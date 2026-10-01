"""Every example must run offline with the mock provider."""

import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCRIPTS = sorted(p for p in EXAMPLES.glob("[0-9]*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_example_runs(
    script: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for key in (
        "SYNAPSI_MODEL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "TAVILY_API_KEY",
        "SYNAPSI_OLLAMA_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(EXAMPLES))
    sys.modules.pop("_common", None)
    runpy.run_path(str(script), run_name="__main__")
    assert capsys.readouterr().out.strip()
