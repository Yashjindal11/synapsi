"""Run several strategies over the same problems and compare them honestly."""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from synapsi._version import __version__
from synapsi.agents.base import Agent
from synapsi.agents.prompts import PROMPT_VERSION
from synapsi.core.problem import Problem
from synapsi.council import Council, JudgeSpec
from synapsi.experiments.metrics import (
    auroc,
    brier,
    expected_calibration_error,
    mcnemar_exact,
    wilson_interval,
)
from synapsi.experiments.problems import ProblemSet
from synapsi.experiments.scoring import Scorer, auto_scorer

CouncilFactory = Callable[[], Council]


def plurality(answers: Sequence[str | None]) -> str | None:
    """Most common non-null answer, or ``None`` on a tie or no answers."""
    counts = Counter(a for a in answers if a is not None).most_common()
    if not counts or (len(counts) > 1 and counts[0][1] == counts[1][1]):
        return None
    return counts[0][0]


class TrialRecord(BaseModel):
    strategy: str
    problem_id: str
    repeat: int
    seed: int
    gold: str | None
    answer: str | None = None
    correct: bool = False
    abstained: bool = True
    verdict: str | None = None
    confidence: float | None = None
    initial_answers: list[str | None] = Field(default_factory=list)
    initial_majority: str | None = None
    initial_majority_correct: bool | None = None
    any_initial_correct: bool | None = None
    initial_agreement: float | None = None
    final_agreement: float | None = None
    position_changes: int = 0
    unsupported_position_changes: int = 0
    unresolved_disagreements: int = 0
    model_calls: int = 0
    tokens: int = 0
    cost_usd: float | None = None
    latency_s: float = 0.0
    run_id: str | None = None
    error: str | None = None


class StrategySummary(BaseModel):
    strategy: str
    trials: int
    errors: int
    accuracy: float
    accuracy_ci: tuple[float, float]
    coverage: float
    selective_accuracy: float | None
    brier: float | None
    ece: float | None
    initial_majority_accuracy: float | None
    right_to_wrong: int
    wrong_to_right: int
    disagreement_error_auroc: float | None
    mean_model_calls: float
    mean_tokens: float
    mean_cost_usd: float | None
    mean_latency_s: float


class Comparison(BaseModel):
    strategy: str
    baseline: str
    pairs: int
    strategy_only_correct: int
    baseline_only_correct: int
    accuracy_delta: float
    mcnemar_p: float


class ExperimentResult(BaseModel):
    name: str
    config: dict[str, Any]
    records: list[TrialRecord]
    summaries: list[StrategySummary]
    comparisons: list[Comparison]
    notes: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"# Experiment: {self.name}", ""]
        lines += [f"> {n}" for n in self.notes]
        if self.notes:
            lines.append("")
        cfg = self.config
        lines += [
            f"- Problems: {cfg['problems']} · repeats: {cfg['repeats']} · mode: {cfg['mode']} · "
            f"seed: {cfg['seed']}",
            f"- SynapSI {cfg['synapsi_version']} · prompts v{cfg['prompt_version']}",
            "",
            "| Strategy | Accuracy (95% CI) | Coverage | Brier | ECE | R→W | W→R | Calls | Tokens "
            "| Cost | Latency |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]

        def f(x: float | None, digits: int = 3) -> str:
            return "—" if x is None else f"{x:.{digits}f}"

        for s in self.summaries:
            lo, hi = s.accuracy_ci
            cost = "unknown" if s.mean_cost_usd is None else f"${s.mean_cost_usd:.4f}"
            lines.append(
                f"| {s.strategy} | {s.accuracy:.3f} ({lo:.2f}-{hi:.2f}) | {s.coverage:.2f} | "
                f"{f(s.brier)} | {f(s.ece)} | {s.right_to_wrong} | {s.wrong_to_right} | "
                f"{s.mean_model_calls:.1f} | {s.mean_tokens:.0f} | {cost} | "
                f"{s.mean_latency_s:.2f}s |"
            )
        if self.comparisons:
            lines += [
                "",
                f"Paired comparison against `{self.comparisons[0].baseline}` (exact McNemar):",
                "",
                "| Strategy | Δ accuracy | only strategy correct | only baseline correct | p |",
                "|---|---|---|---|---|",
            ]
            for c in self.comparisons:
                lines.append(
                    f"| {c.strategy} | {c.accuracy_delta:+.3f} | {c.strategy_only_correct} | "
                    f"{c.baseline_only_correct} | {c.mcnemar_p:.3f} |"
                )
        lines += [
            "",
            "R→W / W→R: problems where the initial independent majority was right and the final "
            "answer wrong, and vice versa. Coverage: share of trials with a committed answer.",
        ]
        return "\n".join(lines) + "\n"

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "records.jsonl").write_text(
            "\n".join(r.model_dump_json() for r in self.records) + "\n", encoding="utf-8"
        )
        summary = self.model_dump(exclude={"records"})
        (d / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        (d / "report.md").write_text(self.to_markdown(), encoding="utf-8")
        return d


def _trial_seed(seed: int, repeat: int, problem_id: str) -> int:
    return int(hashlib.sha256(f"{seed}:{repeat}:{problem_id}".encode()).hexdigest()[:8], 16)


class Experiment:
    """Compare strategies on the same problems with the same agents.

    ``strategies`` is either a list of built-in strategy names (built from
    ``agents`` with identical settings) or a mapping of label to council
    factory for full control (e.g. comparing model mixes or agent counts).
    """

    def __init__(
        self,
        problems: ProblemSet | Sequence[Problem],
        strategies: Sequence[str] | Mapping[str, CouncilFactory],
        *,
        agents: Callable[[], list[Agent]] | None = None,
        judge: JudgeSpec = "auto",
        mode: str = "fast",
        repeats: int = 1,
        seed: int = 0,
        concurrency: int = 4,
        scorer: Scorer = auto_scorer,
        baseline: str | None = None,
        name: str = "experiment",
        save_runs: str | Path | None = None,
    ):
        self.problems = problems if isinstance(problems, ProblemSet) else ProblemSet(problems)
        self.problems.require_answers()
        if isinstance(strategies, Mapping):
            self.factories = dict(strategies)
        else:
            if agents is None:
                raise ValueError("pass `agents` when strategies are given by name")
            make_agents = agents

            def factory(strategy: str) -> CouncilFactory:
                return lambda: Council(make_agents(), strategy=strategy, judge=judge, mode=mode)

            self.factories = {s: factory(s) for s in strategies}
        self.mode = mode
        self.repeats = repeats
        self.seed = seed
        self.concurrency = concurrency
        self.scorer = scorer
        names = list(self.factories)
        self.baseline = baseline if baseline is not None else names[0]
        self.name = name
        self.save_runs = Path(save_runs) if save_runs else None

    async def _trial(
        self, label: str, council: Council, problem: Problem, repeat: int
    ) -> TrialRecord:
        seed = _trial_seed(self.seed, repeat, problem.id)
        record = TrialRecord(
            strategy=label, problem_id=problem.id, repeat=repeat, seed=seed, gold=problem.answer
        )
        start = time.perf_counter()
        try:
            result = await council.run(problem, seed=seed)
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            record.latency_s = time.perf_counter() - start
            return record
        if self.save_runs is not None:
            result.save(self.save_runs / label / f"{problem.id}-r{repeat}.json")
        first = {
            p.agent: p.answer for p in result.perspectives if p.independent and p.stance == "own"
        }
        initial = list(first.values())
        majority = plurality(initial)
        u, usage = result.uncertainty, result.metadata.usage.total
        record.answer = result.answer
        record.abstained = result.answer is None
        record.correct = self.scorer(problem, result.answer)
        record.verdict = u.verdict
        record.confidence = result.judgment.confidence if result.judgment else None
        record.initial_answers = initial
        record.initial_majority = majority
        record.initial_majority_correct = (
            self.scorer(problem, majority) if majority is not None else None
        )
        record.any_initial_correct = any(self.scorer(problem, a) for a in initial if a)
        record.initial_agreement = u.initial_agreement_rate
        record.final_agreement = u.agreement_rate
        record.position_changes = u.position_changes
        record.unsupported_position_changes = u.unsupported_position_changes
        record.unresolved_disagreements = u.unresolved_disagreements
        record.model_calls = usage.calls
        record.tokens = usage.total_tokens
        record.cost_usd = usage.cost_usd
        record.latency_s = result.metadata.latency_s
        record.run_id = result.run_id
        if result.metadata.errors:
            record.error = "; ".join(result.metadata.errors)[:500]
        return record

    async def run(self, progress: Callable[[TrialRecord], None] | None = None) -> ExperimentResult:
        semaphore = asyncio.Semaphore(self.concurrency)
        councils = {label: factory() for label, factory in self.factories.items()}

        async def guarded(label: str, problem: Problem, repeat: int) -> TrialRecord:
            async with semaphore:
                record = await self._trial(label, councils[label], problem, repeat)
            if progress is not None:
                progress(record)
            return record

        records = await asyncio.gather(
            *(
                guarded(label, problem, repeat)
                for label in councils
                for repeat in range(self.repeats)
                for problem in self.problems
            )
        )
        return self._summarise(list(records), councils)

    def run_sync(self, **kwargs: Any) -> ExperimentResult:
        return asyncio.run(self.run(**kwargs))

    def _summarise(
        self, records: list[TrialRecord], councils: dict[str, Council]
    ) -> ExperimentResult:
        summaries = [
            summarise(label, [r for r in records if r.strategy == label]) for label in councils
        ]
        comparisons = []
        base = {(r.problem_id, r.repeat): r for r in records if r.strategy == self.baseline}
        for label in councils:
            if label == self.baseline:
                continue
            pairs = [
                (base[(r.problem_id, r.repeat)], r)
                for r in records
                if r.strategy == label and (r.problem_id, r.repeat) in base
            ]
            b = sum(1 for x, y in pairs if x.correct and not y.correct)
            c = sum(1 for x, y in pairs if y.correct and not x.correct)
            comparisons.append(
                Comparison(
                    strategy=label,
                    baseline=self.baseline,
                    pairs=len(pairs),
                    strategy_only_correct=c,
                    baseline_only_correct=b,
                    accuracy_delta=(c - b) / len(pairs) if pairs else 0.0,
                    mcnemar_p=mcnemar_exact(b, c),
                )
            )
        models = sorted({a["model"] for c in councils.values() for a in c.describe()["agents"]})
        notes = []
        if all(m.startswith("mock:") for m in models):
            notes.append(
                "All agents used the mock provider: these numbers exercise the pipeline and "
                "carry no information about reasoning quality."
            )
        if any(s.mean_cost_usd is None for s in summaries):
            notes.append("Cost is unknown for some models because no pricing was configured.")
        return ExperimentResult(
            name=self.name,
            config={
                "problems": len(self.problems),
                "problem_set": self.problems.name,
                "repeats": self.repeats,
                "seed": self.seed,
                "mode": self.mode,
                "baseline": self.baseline,
                "councils": {label: c.describe() for label, c in councils.items()},
                "models": models,
                "synapsi_version": __version__,
                "prompt_version": PROMPT_VERSION,
                "python": platform.python_version(),
                "created_at": datetime.now(UTC).isoformat(),
            },
            records=records,
            summaries=summaries,
            comparisons=comparisons,
            notes=notes,
        )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(label: str, records: list[TrialRecord]) -> StrategySummary:
    n = len(records)
    correct = sum(r.correct for r in records)
    answered = [r for r in records if not r.abstained]
    calibrated = [r for r in answered if r.confidence is not None]
    initial_known = [r for r in records if r.initial_majority_correct is not None]
    disagreement = [r for r in records if r.initial_agreement is not None]
    costs = [r.cost_usd for r in records]
    return StrategySummary(
        strategy=label,
        trials=n,
        errors=sum(1 for r in records if r.error),
        accuracy=correct / n if n else 0.0,
        accuracy_ci=wilson_interval(correct, n),
        coverage=len(answered) / n if n else 0.0,
        selective_accuracy=sum(r.correct for r in answered) / len(answered) if answered else None,
        brier=brier([r.confidence or 0.0 for r in calibrated], [r.correct for r in calibrated]),
        ece=expected_calibration_error(
            [r.confidence or 0.0 for r in calibrated], [r.correct for r in calibrated]
        ),
        initial_majority_accuracy=(
            sum(bool(r.initial_majority_correct) for r in initial_known) / len(initial_known)
            if initial_known
            else None
        ),
        right_to_wrong=sum(1 for r in records if r.initial_majority_correct and not r.correct),
        wrong_to_right=sum(1 for r in records if r.initial_majority_correct is False and r.correct),
        disagreement_error_auroc=auroc(
            [1 - (r.initial_agreement or 0.0) for r in disagreement],
            [not r.correct for r in disagreement],
        ),
        mean_model_calls=_mean([r.model_calls for r in records]),
        mean_tokens=_mean([r.tokens for r in records]),
        mean_cost_usd=None if any(c is None for c in costs) else _mean([c or 0.0 for c in costs]),
        mean_latency_s=_mean([r.latency_s for r in records]),
    )
