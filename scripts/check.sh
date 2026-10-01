#!/usr/bin/env bash
# Local quality gate: mirrors CI.
set -euo pipefail
ruff format --check .
ruff check .
mypy
pytest
