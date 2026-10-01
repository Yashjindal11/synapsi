"""``synapsi`` command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from synapsi import __version__
from synapsi.agents.base import Agent
from synapsi.agents.roles import list_roles
from synapsi.config import DEFAULT_CONFIG, build_council, load_config
from synapsi.core.errors import ConfigError, SynapSIError
from synapsi.core.problem import Problem
from synapsi.council import Council
from synapsi.experiments import SUITES, Experiment, ExperimentResult, ProblemSet, TrialRecord
from synapsi.experiments.engine import CouncilFactory
from synapsi.observability.events import Event, EventType
from synapsi.providers.registry import PROVIDER_KEY_ENV, available_providers
from synapsi.result import SynapSIResult
from synapsi.strategies import list_strategies

ENV_EXAMPLE = """\
# Copy to .env (never commit it) or export these variables. Only set what you use.
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
HF_TOKEN=
TAVILY_API_KEY=
BRAVE_API_KEY=
"""

DEFAULT_CONFIG_FILES = ("synapsi.yaml", "synapsi.yml", "synapsi.toml", "synapsi.json")


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _read_arg(value: str | None) -> str | None:
    """``@path`` reads a file; anything else is literal text."""
    if value and value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


def _find_config(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit)
    for name in DEFAULT_CONFIG_FILES:
        if Path(name).exists():
            return Path(name)
    return None


def _council(args: argparse.Namespace, strategy: str | None = None) -> Council:
    path = _find_config(getattr(args, "config", None))
    strategy = strategy or getattr(args, "strategy", None)
    mode = getattr(args, "mode", None)
    if path is not None:
        return build_council(
            load_config(path),
            base_dir=path.parent,
            model_override=args.model,
            strategy_override=strategy,
            mode_override=mode,
        )
    model = args.model or "mock"
    if model == "mock":
        _err("note: no config and no --model given; using the mock provider (placeholder output).")
    roles = [r.strip() for r in args.roles.split(",")] if getattr(args, "roles", None) else None
    if roles:
        council = Council(
            [Agent.from_role(r, model) for r in roles],
            strategy=strategy or "independent_panel",
            mode=mode or "balanced",
        )
    else:
        council = Council.preset(mode or "fast", model, strategy=strategy or "independent_panel")
    return council


def _progress(event: Event) -> None:
    interesting = {
        EventType.STEP_STARTED: lambda e: f"  step {e.step}",
        EventType.AGENT_COMPLETED: lambda e: (
            f"    {e.agent}"
            + (" [adversarial]" if e.data.get("stance") == "adversarial" else "")
            + f": answer={e.data.get('answer')}"
        ),
        EventType.AGENT_FAILED: lambda e: f"    {e.agent} FAILED: {e.data.get('error')}",
        EventType.JUDGMENT_CREATED: lambda e: (
            f"  judgment: {e.data.get('verdict')} {e.data.get('answer') or ''}"
        ),
        EventType.WARNING: lambda e: f"  warning: {e.data.get('message')}",
    }
    fmt = interesting.get(event.type)
    if fmt is not None:
        _err(fmt(event))


def _render(result: SynapSIResult, fmt: str) -> str:
    if fmt == "json":
        return result.to_json()
    if fmt == "html":
        return result.to_html()
    if fmt == "md":
        return result.to_markdown()
    return _summary(result)


def _summary(result: SynapSIResult) -> str:
    j, s, u = result.judgment, result.synthesis, result.metadata.usage.total
    lines = [f"Question: {result.problem.question}"]
    if j:
        lines.append(
            f"Verdict:  {j.verdict.value}"
            + (f"  answer={j.answer}" if j.answer else "")
            + f"  confidence={j.confidence:.2f}  evidence={j.evidence_strength.value}"
        )
        lines.append(f"Decision: {j.decision}")
    if s:
        lines.append(f"Summary:  {s.summary}")
        for title, items in (
            ("Established", s.established),
            ("Probable", s.probable),
            ("Disputed", s.disputed),
            ("Unknown", s.unknown),
        ):
            lines.append(f"{title} ({len(items)}):")
            lines += [f"  - {f.statement}" for f in items[:5]]
    cost = "unknown" if u.cost_usd is None else f"${u.cost_usd:.4f}"
    lines.append(
        f"Usage:    {u.calls} calls, {u.total_tokens} tokens, cost {cost}, "
        f"{result.metadata.latency_s:.1f}s  (run {result.run_id})"
    )
    return "\n".join(lines)


def _emit(text: str, output: str | None) -> None:
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text, encoding="utf-8")
        _err(f"wrote {output}")
    else:
        print(text)


# -- commands -------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in (("synapsi.yaml", DEFAULT_CONFIG), (".env.example", ENV_EXAMPLE)):
        path = directory / name
        if path.exists() and not args.force:
            _err(f"skip {path} (exists; use --force to overwrite)")
            continue
        path.write_text(content, encoding="utf-8")
        _err(f"created {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    council = _council(args)
    problem = Problem(
        question=args.question,
        context=_read_arg(args.context),
        options=args.option or None,
        facts=[f for f in (_read_arg(x) for x in args.fact or []) if f],
    )
    handlers = [] if args.quiet else [_progress]
    result = asyncio.run(council.run(problem, seed=args.seed, event_handlers=handlers))
    if args.save_dir:
        path = result.save(Path(args.save_dir) / f"{result.run_id}.json")
        _err(f"saved {path}")
    _emit(_render(result, args.format), args.output)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    result = SynapSIResult.load(args.result)
    section = args.section
    if section == "summary":
        print(_summary(result))
    elif section == "claims":
        for c in result.claims:
            flag = " (withdrawn)" if c.withdrawn else ""
            print(f"{c.id:>4} [{c.status.value:<19}] {c.agent}: {c.statement}{flag}")
    elif section == "evidence":
        for e in result.evidence:
            src = e.provenance.source_id or e.provenance.claimed_source or ""
            print(f"{e.id:>4} [{e.provenance.source_kind.value}] {e.content[:120]} {src}")
    elif section == "perspectives":
        for p in result.perspectives:
            kind = "independent" if p.independent else f"round {p.round}"
            print(
                f"{p.agent} ({p.role}, {kind}, {p.stance}): answer={p.answer} "
                f"conf={p.confidence:.2f}\n    {p.position[:200]}"
            )
    elif section == "challenges":
        for ch in result.challenges:
            print(
                f"{ch.id} {ch.challenger} -> {ch.target_claim_id} [{ch.status.value}]: {ch.problem}"
            )
    elif section == "disagreements":
        for d in result.disagreements:
            print(f"{d.id} [{d.kind}, {d.status}] {d.topic} :: {d.evidence_balance}")
    elif section == "usage":
        print(result.metadata.usage.model_dump_json(indent=2))
    elif section == "uncertainty":
        print(result.uncertainty.model_dump_json(indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    result = SynapSIResult.load(args.result)
    _emit(_render(result, args.format), args.output)
    return 0


def cmd_providers(_: argparse.Namespace) -> int:
    for name, description in sorted(available_providers().items()):
        env = PROVIDER_KEY_ENV.get(name)
        status = (
            ""
            if env is None
            else (f"  [{env}: set]" if os.environ.get(env) else f"  [{env}: not set]")
        )
        print(f"{name:<18} {description}{status}")
    return 0


def cmd_workflows(_: argparse.Namespace) -> int:
    for name, description in list_strategies().items():
        print(f"{name:<20} {description}")
    return 0


def cmd_agents(_: argparse.Namespace) -> int:
    for role in list_roles():
        stance = " (contrarian)" if role.stance == "contrarian" else ""
        print(f"{role.key:<18} {role.description}{stance}")
    return 0


def _experiment(args: argparse.Namespace, problems: ProblemSet, name: str) -> int:
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    def factory(strategy: str) -> CouncilFactory:
        return lambda: _council(args, strategy)

    factories = {s: factory(s) for s in strategies}
    done = 0
    total = len(strategies) * len(problems) * args.repeats

    def progress(record: TrialRecord) -> None:
        nonlocal done
        done += 1
        mark = "ok " if record.correct else ("ERR" if record.error else "x  ")
        _err(f"[{done}/{total}] {mark} {record.strategy:<20} {record.problem_id}")

    exp = Experiment(
        problems,
        factories,
        repeats=args.repeats,
        seed=args.seed,
        concurrency=args.concurrency,
        baseline=args.baseline,
        mode=args.mode or "fast",
        name=name,
        save_runs=Path(args.out) / "runs" if args.out and args.save_runs else None,
    )
    result: ExperimentResult = asyncio.run(exp.run(progress=None if args.quiet else progress))
    if args.out:
        result.save(args.out)
        _err(f"saved {args.out}/summary.json, records.jsonl, report.md")
    print(result.to_markdown())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    problems = ProblemSet.load(args.problems)
    return _experiment(args, problems, name=problems.name)


def cmd_benchmark(args: argparse.Namespace) -> int:
    if args.suite not in SUITES:
        raise ConfigError(f"unknown suite {args.suite!r}; choose from {', '.join(SUITES)}")
    problems = SUITES[args.suite](n=args.n, seed=args.seed)
    return _experiment(args, problems, name=f"benchmark-{problems.name}")


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from synapsi.server import create_app
    except ImportError:
        _err("the web server needs extras: pip install 'synapsi[server]'")
        return 2
    static = Path(args.static) if args.static else Path("web/frontend/dist")
    app = create_app(
        config_path=_find_config(args.config),
        runs_dir=Path(args.runs_dir),
        static_dir=static if static.exists() else None,
    )
    if args.host not in ("127.0.0.1", "localhost"):
        _err("warning: the API has no user accounts; set SYNAPSI_API_TOKEN before exposing it.")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


# -- parser ---------------------------------------------------------------------


def _add_council_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", help="config file (default: ./synapsi.yaml if present)")
    p.add_argument("--model", help="model spec for all agents, e.g. openai:gpt-4o-mini")
    p.add_argument("--mode", choices=["fast", "balanced", "deep", "custom"])
    p.add_argument("--roles", help="comma-separated roles when no config is used")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synapsi", description="Structured multi-agent deliberation and evaluation."
    )
    parser.add_argument("--version", action="version", version=f"synapsi {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write a starter synapsi.yaml and .env.example")
    p.add_argument("dir", nargs="?", default=".")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    for name in ("run", "deliberate"):
        p = sub.add_parser(name, help="deliberate on a question")
        p.add_argument("question")
        _add_council_args(p)
        p.add_argument("-s", "--strategy", help="see `synapsi workflows`")
        p.add_argument("--option", action="append", help="answer option (repeatable)")
        p.add_argument("--context", help="background text or @file")
        p.add_argument("--fact", action="append", help="user-provided fact or @file (repeatable)")
        p.add_argument("--seed", type=int)
        p.add_argument(
            "-f", "--format", choices=["summary", "md", "json", "html"], default="summary"
        )
        p.add_argument("-o", "--output", help="write output to a file")
        p.add_argument("--save-dir", default=None, help="also save full JSON result here")
        p.set_defaults(func=cmd_run)

    p = sub.add_parser("inspect", help="inspect a saved result JSON")
    p.add_argument("result")
    p.add_argument(
        "--section",
        default="summary",
        choices=[
            "summary",
            "claims",
            "evidence",
            "perspectives",
            "challenges",
            "disagreements",
            "uncertainty",
            "usage",
        ],
    )
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("report", help="render a saved result as Markdown/HTML/JSON")
    p.add_argument("result")
    p.add_argument("-f", "--format", choices=["md", "html", "json", "summary"], default="md")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_report)

    sub.add_parser("providers", help="list model providers and key status").set_defaults(
        func=cmd_providers
    )
    sub.add_parser("workflows", help="list built-in strategies").set_defaults(func=cmd_workflows)
    sub.add_parser("agents", help="list built-in roles").set_defaults(func=cmd_agents)

    for name, helptext in (
        ("evaluate", "compare strategies on a problem file (.jsonl/.json/.yaml)"),
        ("benchmark", "compare strategies on a built-in synthetic suite"),
    ):
        p = sub.add_parser(name, help=helptext)
        if name == "evaluate":
            p.add_argument("problems")
        else:
            p.add_argument("--suite", default="base_rates", help=f"one of: {', '.join(SUITES)}")
            p.add_argument("-n", type=int, default=20, help="number of problems")
        _add_council_args(p)
        p.add_argument(
            "--strategies", default="single_model,majority_vote,independent_panel,debate"
        )
        p.add_argument("--baseline", help="strategy to compare against (default: first)")
        p.add_argument("--repeats", type=int, default=1)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--concurrency", type=int, default=4)
        p.add_argument("--out", help="directory for summary.json, records.jsonl, report.md")
        p.add_argument("--save-runs", action="store_true", help="also save every run's JSON")
        p.set_defaults(func=cmd_evaluate if name == "evaluate" else cmd_benchmark)

    p = sub.add_parser("serve", help="start the web API and dashboard")
    p.add_argument("-c", "--config")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--static", help="built dashboard directory (default: web/frontend/dist)")
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
        return code
    except (ConfigError, FileNotFoundError) as exc:
        _err(f"error: {exc}")
        return 2
    except SynapSIError as exc:
        _err(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
