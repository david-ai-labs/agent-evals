"""Regular-expression matching with a hard time limit.

Python's ``re`` engine backtracks, so a pattern such as ``(a+)+$`` can take
exponential time on a short input. ``re`` offers no timeout, and a search holds
the interpreter lock for its whole duration, so it cannot be bounded from
another thread either. Patterns come from case files, which may not be written
by the person running the evaluation, so each search runs in a short-lived
child interpreter that is killed if it overruns.

This costs one process start per ``matches`` check (tens of milliseconds). It
does not affect measured agent latency, which is taken around the adapter call
only.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass

DEFAULT_REGEX_TIMEOUT_SECONDS = 2.0
MAX_PATTERN_LENGTH = 1000

# Reads {"pattern", "text", "flags"} as JSON on stdin; prints 1 or 0.
_CHILD = (
    "import json,re,sys;"
    "j=json.load(sys.stdin);"
    "sys.stdout.write('1' if re.search(j['pattern'],j['text'],j['flags']) else '0')"
)


@dataclass(frozen=True, slots=True)
class RegexOutcome:
    matched: bool | None
    """``None`` when the search could not be completed."""
    problem: str = ""
    """Why the search could not be completed; empty when ``matched`` is set."""


def pattern_problem(pattern: str) -> str | None:
    """Explain why ``pattern`` is unacceptable, or return ``None`` if it is fine."""
    if len(pattern) > MAX_PATTERN_LENGTH:
        return f"pattern is {len(pattern)} characters long (limit {MAX_PATTERN_LENGTH})"
    try:
        re.compile(pattern)
    except re.error as exc:
        return f"not a valid regular expression: {exc}"
    except (RecursionError, OverflowError) as exc:
        return f"pattern is too complex to compile ({type(exc).__name__})"
    return None


def bounded_search(
    pattern: str,
    text: str,
    *,
    ignore_case: bool = False,
    timeout_seconds: float = DEFAULT_REGEX_TIMEOUT_SECONDS,
) -> RegexOutcome:
    """``re.search`` that gives up, and says so, after ``timeout_seconds``."""
    if not sys.executable:
        return RegexOutcome(None, "no Python executable available to run a bounded search")
    payload = json.dumps(
        {"pattern": pattern, "text": text, "flags": int(re.IGNORECASE) if ignore_case else 0}
    )
    try:
        # -I: isolated from the environment and working directory. -S: skip site.
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _CHILD],
            input=payload.encode("ascii"),
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return RegexOutcome(None, f"search did not finish within {timeout_seconds:g}s")
    except OSError as exc:
        return RegexOutcome(None, f"could not start the search process: {exc}")
    if completed.returncode != 0 or completed.stdout not in (b"0", b"1"):
        return RegexOutcome(None, f"search process failed (exit status {completed.returncode})")
    return RegexOutcome(completed.stdout == b"1")
