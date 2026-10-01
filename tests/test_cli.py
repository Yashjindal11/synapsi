import json
from pathlib import Path

import pytest

from synapsi.cli.main import main


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def test_listing_commands(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY: set" in out and "sk-secret-value" not in out
    assert main(["workflows"]) == 0
    assert "adversarial" in capsys.readouterr().out
    assert main(["agents"]) == 0
    assert "devils_advocate" in capsys.readouterr().out


def test_init_run_inspect_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["init"]) == 0
    assert (tmp_path / "synapsi.yaml").exists() and (tmp_path / ".env.example").exists()
    code = main(
        [
            "run",
            "Should we migrate to service X?",
            "--option",
            "yes",
            "--option",
            "no",
            "--fact",
            "Service X costs $2k/month.",
            "--format",
            "json",
            "--save-dir",
            "runs",
            "-q",
        ]
    )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["metadata"]["strategy"] == "debate"
    saved = next((tmp_path / "runs").glob("*.json"))
    for section in ("summary", "claims", "evidence", "uncertainty"):
        assert main(["inspect", str(saved), "--section", section]) == 0
    assert main(["report", str(saved), "-f", "html", "-o", "r.html"]) == 0
    assert (tmp_path / "r.html").read_text().startswith("<!doctype html>")


def test_run_without_config_uses_mock_preset(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "q?", "--roles", "skeptic,statistician", "-q"]) == 0
    captured = capsys.readouterr()
    assert "mock provider" in captured.err
    assert "Verdict:" in captured.out


def test_benchmark_and_evaluate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "benchmark",
            "--suite",
            "ordering",
            "-n",
            "2",
            "--strategies",
            "single_model,majority_vote",
            "--out",
            "bench",
            "-q",
        ]
    )
    assert code == 0
    assert "| majority_vote |" in capsys.readouterr().out
    assert (tmp_path / "bench" / "records.jsonl").exists()
    problems = tmp_path / "p.jsonl"
    problems.write_text('{"question": "2+2?", "answer": "4"}\n')
    assert main(["evaluate", str(problems), "--strategies", "single_model", "-q"]) == 0


def test_config_errors_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", "q", "-c", "missing.yaml"]) == 2
    assert "not found" in capsys.readouterr().err
    assert main(["benchmark", "--suite", "nope"]) == 2
