from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from david_agent_evals import EvalCase, EvalSuite, parse_suite

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


def make_case(expect: dict[str, Any] | None = None, **fields: Any) -> EvalCase:
    """Build one validated case through the same parser the case files use."""
    raw: dict[str, Any] = {"id": "case", "input": "hello", **fields}
    if expect is not None:
        raw["expect"] = expect
    return parse_suite({"cases": [raw]}).cases[0]


def make_suite(*cases: dict[str, Any], **fields: Any) -> EvalSuite:
    return parse_suite({"name": "test-suite", "cases": list(cases), **fields})


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES
