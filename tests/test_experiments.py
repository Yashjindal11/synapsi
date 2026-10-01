from pathlib import Path
from typing import Any

import pytest

from synapsi import Agent, Council
from synapsi.core import Problem
from synapsi.core.errors import ConfigError
from synapsi.experiments import SUITES, Experiment, ProblemSet, auto_scorer
from synapsi.experiments.engine import plurality
from synapsi.experiments.metrics import (
    auroc,
    brier,
    expected_calibration_error,
    mcnemar_exact,
    wilson_interval,
)
from synapsi.providers import CompletionRequest, MockProvider
from synapsi.testing import scripted


def test_metrics() -> None:
    lo, hi = wilson_interval(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.97
    assert brier([1.0, 0.0], [True, False]) == 0.0
    assert expected_calibration_error([0.9, 0.9], [True, False]) == pytest.approx(0.4)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert auroc([0.9, 0.1], [True, False]) == 1.0
    assert auroc([0.5], [True]) is None
    assert plurality(["a", "b", "a", None]) == "a"
    assert plurality(["a", "b"]) is None


@pytest.mark.parametrize("suite", sorted(SUITES))
def test_suites_are_deterministic_and_answered(suite: str) -> None:
    a, b = SUITES[suite](n=5, seed=3), SUITES[suite](n=5, seed=3)
    assert [p.model_dump() for p in a] == [p.model_dump() for p in b]
    assert all(p.answer for p in a)
    assert all(p.answer in p.options for p in a if p.options)


def test_scoring() -> None:
    num = Problem(question="q", answer="1250")
    assert auto_scorer(num, "There are 1,250 boxes.")
    assert not auto_scorer(num, "1251")
    mc = Problem(question="q", options=["yes", "no"], answer="yes")
    assert auto_scorer(mc, "Yes")
    assert not auto_scorer(mc, None)


def test_problem_set_roundtrip(tmp_path: Path) -> None:
    ps = SUITES["ordering"](n=3, seed=1)
    loaded = ProblemSet.load(ps.save(tmp_path / "p.jsonl"))
    assert [p.answer for p in loaded] == [p.answer for p in ps]
    with pytest.raises(ConfigError):
        ProblemSet([Problem(question="q")]).require_answers()


def _oracle(correct: bool) -> MockProvider:
    """Answers each problem right (or wrong) using the gold label smuggled via a closure."""
    golds = {p.question: p.answer for p in SUITES["base_rates"](n=6, seed=0)}

    def script(req: CompletionRequest) -> dict[str, Any] | None:
        if req.metadata["task"] in ("analyze", "revise"):
            gold = golds[req.metadata["question"]]
            answer = gold if correct else ("no" if gold == "yes" else "yes")
            return {"position": answer, "answer": answer, "confidence": 0.8}
        return None

    return scripted(script)


async def test_experiment_compares_strategies_and_detects_difference(tmp_path: Path) -> None:
    problems = SUITES["base_rates"](n=6, seed=0)
    exp = Experiment(
        problems,
        {
            "right": lambda: Council([Agent("a", model=_oracle(True))], strategy="single_model"),
            "wrong": lambda: Council([Agent("a", model=_oracle(False))], strategy="single_model"),
        },
        repeats=2,
        name="oracle",
        save_runs=tmp_path / "runs",
    )
    result = await exp.run()
    by = {s.strategy: s for s in result.summaries}
    assert by["right"].accuracy == 1.0 and by["wrong"].accuracy == 0.0
    assert by["right"].coverage == 1.0
    [cmp] = result.comparisons
    assert cmp.baseline == "right" and cmp.baseline_only_correct == 12
    assert cmp.mcnemar_p < 0.001
    out = result.save(tmp_path / "exp")
    assert "| right |" in (out / "report.md").read_text()
    assert len(list((tmp_path / "runs" / "right").glob("*.json"))) == 12


async def test_named_strategies_share_agents_and_flag_mock() -> None:
    exp = Experiment(
        SUITES["ordering"](n=2, seed=0),
        ["single_model", "majority_vote", "debate"],
        agents=lambda: [Agent(n, model=MockProvider()) for n in ("a", "b", "c")],
        judge="structural",
    )
    result = await exp.run()
    assert [s.strategy for s in result.summaries] == ["single_model", "majority_vote", "debate"]
    assert any("mock provider" in n for n in result.notes)
    assert all(r.error is None for r in result.records)
    assert result.config["prompt_version"]
