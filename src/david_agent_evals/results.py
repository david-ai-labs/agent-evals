"""Result records produced by the runner, and their aggregate summary."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from david_agent_evals.metrics import MetricResult
from david_agent_evals.types import ToolCall, Usage


class CaseStatus(StrEnum):
    PASSED = "passed"
    """The agent responded and every evaluated metric passed."""
    FAILED = "failed"
    """The agent responded but at least one metric failed."""
    ERROR = "error"
    """The adapter raised an exception or returned an invalid response."""
    TIMEOUT = "timeout"
    """The adapter did not respond within the case's timeout."""


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    status: CaseStatus
    latency_seconds: float
    timeout_seconds: float
    description: str = ""
    tags: tuple[str, ...] = ()
    output: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = None
    cost_usd: float | None = None
    metrics: tuple[MetricResult, ...] = ()
    error: str | None = None

    @property
    def responded(self) -> bool:
        return self.status in (CaseStatus.PASSED, CaseStatus.FAILED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "description": self.description,
            "tags": list(self.tags),
            "status": self.status.value,
            "latency_seconds": self.latency_seconds,
            "timeout_seconds": self.timeout_seconds,
            "output": self.output,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "usage": self.usage.to_dict() if self.usage else None,
            "cost_usd": self.cost_usd,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """One metric summarised across the cases where it produced a result."""

    name: str
    evaluated: int
    """Cases where the metric gave a pass/fail verdict."""
    passed: int
    measured: int
    """Cases where the metric produced a value."""
    mean_value: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "evaluated": self.evaluated,
            "passed": self.passed,
            "measured": self.measured,
            "mean_value": self.mean_value,
        }


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Latency over the cases where the agent responded. Timed-out and
    errored cases are excluded because their duration is not a response time."""

    count: int
    mean: float
    p50: float
    p95: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_seconds": self.mean,
            "p50_seconds": self.p50,
            "p95_seconds": self.p95,
            "max_seconds": self.max,
        }


@dataclass(frozen=True, slots=True)
class SuiteSummary:
    total: int
    passed: int
    failed: int
    errors: int
    timeouts: int
    metrics: tuple[MetricAggregate, ...]
    latency: LatencyStats | None
    total_cost_usd: float | None
    """Sum over the cases with a known cost; ``None`` if no case had one."""
    cases_with_cost: int

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "timeouts": self.timeouts,
            "pass_rate": self.pass_rate,
            "metrics": [aggregate.to_dict() for aggregate in self.metrics],
            "latency": self.latency.to_dict() if self.latency else None,
            "total_cost_usd": self.total_cost_usd,
            "cases_with_cost": self.cases_with_cost,
        }


@dataclass(frozen=True, slots=True)
class SuiteResult:
    suite_name: str
    agent_name: str
    started_at: str
    """UTC timestamp, ISO 8601."""
    duration_seconds: float
    cases: tuple[CaseResult, ...]
    source: str | None = None

    @property
    def all_passed(self) -> bool:
        return all(case.status is CaseStatus.PASSED for case in self.cases)

    @property
    def summary(self) -> SuiteSummary:
        return summarize(self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite_name,
            "source": self.source,
            "agent": self.agent_name,
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
            "summary": self.summary.to_dict(),
            "cases": [case.to_dict() for case in self.cases],
        }


def summarize(cases: Sequence[CaseResult]) -> SuiteSummary:
    def count(status: CaseStatus) -> int:
        return sum(case.status is status for case in cases)

    by_metric: dict[str, list[MetricResult]] = {}
    for case in cases:
        for metric in case.metrics:
            by_metric.setdefault(metric.name, []).append(metric)
    aggregates = []
    for name, results in by_metric.items():
        values = [r.value for r in results if r.value is not None]
        aggregates.append(
            MetricAggregate(
                name=name,
                evaluated=sum(r.passed is not None for r in results),
                passed=sum(r.passed is True for r in results),
                measured=len(values),
                mean_value=sum(values) / len(values) if values else None,
            )
        )

    latencies = sorted(case.latency_seconds for case in cases if case.responded)
    latency = (
        LatencyStats(
            count=len(latencies),
            mean=sum(latencies) / len(latencies),
            p50=percentile(latencies, 50),
            p95=percentile(latencies, 95),
            max=latencies[-1],
        )
        if latencies
        else None
    )
    costs = [case.cost_usd for case in cases if case.cost_usd is not None]
    return SuiteSummary(
        total=len(cases),
        passed=count(CaseStatus.PASSED),
        failed=count(CaseStatus.FAILED),
        errors=count(CaseStatus.ERROR),
        timeouts=count(CaseStatus.TIMEOUT),
        metrics=tuple(aggregates),
        latency=latency,
        total_cost_usd=sum(costs) if costs else None,
        cases_with_cost=len(costs),
    )


def percentile(sorted_values: Sequence[float], percent: float) -> float:
    """Nearest-rank percentile of an ascending, non-empty sequence."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence is undefined")
    rank = max(1, math.ceil(percent / 100 * len(sorted_values)))
    return sorted_values[rank - 1]
