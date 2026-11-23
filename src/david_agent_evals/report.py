"""Human-readable and JSON reports for completed evaluation runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import david_agent_evals
from david_agent_evals.results import CaseResult, CaseStatus, SuiteResult

SCHEMA_VERSION = "0.1"

_LABELS = {
    CaseStatus.PASSED: "PASS",
    CaseStatus.FAILED: "FAIL",
    CaseStatus.ERROR: "ERROR",
    CaseStatus.TIMEOUT: "TIMEOUT",
}


def build_report(results: Sequence[SuiteResult]) -> dict[str, Any]:
    """The JSON document written by ``--output`` and ``--format json``."""
    return {
        "schema_version": SCHEMA_VERSION,
        "framework_version": david_agent_evals.__version__,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "all_passed": all(result.all_passed for result in results),
        "suites": [result.to_dict() for result in results],
    }


def to_json(results: Sequence[SuiteResult]) -> str:
    # default=str keeps the report writable if an adapter put a non-JSON
    # value (a datetime, say) into tool-call arguments.
    return json.dumps(build_report(results), indent=2, default=str)


def write_json(results: Sequence[SuiteResult], path: str | Path) -> None:
    Path(path).write_text(to_json(results) + "\n", encoding="utf-8")


def render_text(results: Sequence[SuiteResult]) -> str:
    """A plain-text report: one line per case, failure reasons, and a summary."""
    return "\n\n".join(_render_suite(result) for result in results)


def _render_suite(result: SuiteResult) -> str:
    lines = [
        f"DAVID Agent Evaluations v{david_agent_evals.__version__}",
        f"Suite: {result.suite_name}" + (f"  ({result.source})" if result.source else ""),
        f"Agent: {result.agent_name}",
        "",
    ]
    width = max((len(case.case_id) for case in result.cases), default=0)
    for case in result.cases:
        cost = f"${case.cost_usd:.6f}" if case.cost_usd is not None else "-"
        lines.append(
            f"  {_LABELS[case.status]:<7}  {case.case_id:<{width}}  "
            f"{_seconds(case.latency_seconds):>9}  {cost:>10}"
        )
        lines.extend(f"           {reason}" for reason in _reasons(case))

    summary = result.summary
    lines += [
        "",
        "Summary",
        f"  Cases       {summary.total} total: {summary.passed} passed, {summary.failed} failed, "
        f"{summary.errors} errors, {summary.timeouts} timeouts",
        f"  Pass rate   {summary.pass_rate:.1%}",
    ]
    if summary.latency is not None:
        stats = summary.latency
        lines.append(
            f"  Latency     mean {_seconds(stats.mean)}, p50 {_seconds(stats.p50)}, "
            f"p95 {_seconds(stats.p95)}, max {_seconds(stats.max)} "
            f"({stats.count} responding case(s))"
        )
    if summary.total_cost_usd is not None:
        lines.append(
            f"  Cost        ${summary.total_cost_usd:.6f} total "
            f"({summary.cases_with_cost} of {summary.total} case(s) reported usage)"
        )
    else:
        lines.append("  Cost        not available (no case reported usable usage data)")
    lines.append(f"  Duration    {_seconds(result.duration_seconds)}")

    checks = [aggregate for aggregate in summary.metrics if aggregate.evaluated]
    if checks:
        lines += ["", "Checks (over the cases where the agent responded and the check applied)"]
    for aggregate in checks:
        line = f"  {aggregate.name:<21}  {aggregate.passed}/{aggregate.evaluated} passed"
        if aggregate.name == "tool_call_correctness" and aggregate.mean_value is not None:
            line += f", mean score {aggregate.mean_value:.2f}"
        lines.append(line)
    return "\n".join(lines)


def _reasons(case: CaseResult) -> list[str]:
    if case.error is not None:
        return [case.error]
    return [f"{metric.name}: {metric.detail}" for metric in case.metrics if metric.passed is False]


def _seconds(value: float) -> str:
    return f"{value * 1000:.2f}ms" if value < 1 else f"{value:.2f}s"
