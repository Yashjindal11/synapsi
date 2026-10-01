"""Problem sets: load, save, and validate collections of problems with gold answers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from synapsi.core.errors import ConfigError
from synapsi.core.problem import Problem


class ProblemSet:
    def __init__(self, problems: Iterable[Problem], name: str = "problems"):
        self.problems = list(problems)
        self.name = name
        ids = [p.id for p in self.problems]
        if len(set(ids)) != len(ids):
            raise ConfigError(f"problem ids must be unique in {name}")

    def __iter__(self) -> Iterator[Problem]:
        return iter(self.problems)

    def __len__(self) -> int:
        return len(self.problems)

    def require_answers(self) -> None:
        missing = [p.id for p in self.problems if p.answer is None]
        if missing:
            raise ConfigError(f"{len(missing)} problems have no gold answer, e.g. {missing[:3]}")

    @classmethod
    def load(cls, path: str | Path) -> ProblemSet:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        rows: Any
        if p.suffix == ".jsonl":
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif p.suffix == ".json":
            rows = json.loads(text)
        elif p.suffix in (".yaml", ".yml"):
            import yaml

            rows = yaml.safe_load(text)
        else:
            raise ConfigError(f"unsupported problem file {p.suffix!r}")
        if isinstance(rows, dict):
            rows = rows.get("problems", [])
        return cls((Problem.model_validate(r) for r in rows), name=p.stem)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            "\n".join(pr.model_dump_json(exclude_defaults=True) for pr in self.problems) + "\n",
            encoding="utf-8",
        )
        return p
