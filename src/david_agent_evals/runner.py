"""Runs evaluation cases against an agent adapter and scores the responses."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from david_agent_evals.adapters import AgentAdapter
from david_agent_evals.cases import EvalCase, EvalSuite, Pricing
from david_agent_evals.metrics import (
    Metric,
    MetricContext,
    MetricResult,
    default_metrics,
    resolve_cost,
)
from david_agent_evals.results import CaseResult, CaseStatus, SuiteResult
from david_agent_evals.types import AgentRequest, AgentResponse

DEFAULT_TIMEOUT_SECONDS = 30.0


class EvalRunner:
    """Runs cases one at a time against a single adapter.

    Timeout precedence, most specific first: the case's ``timeout_seconds``,
    the suite's ``defaults.timeout_seconds``, then ``default_timeout_seconds``.

    Each adapter call runs on its own daemon thread so a hung agent cannot
    stall the run. An ``async`` adapter is cancelled when its timeout expires.
    A blocking adapter cannot be interrupted from outside: it is abandoned and
    reported as a timeout, and its thread ends when the call eventually
    returns or the process exits.
    """

    def __init__(
        self,
        adapter: AgentAdapter,
        *,
        default_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        metrics: Sequence[Metric] | None = None,
    ) -> None:
        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be greater than 0")
        self.adapter = adapter
        self.default_timeout_seconds = default_timeout_seconds
        self.metrics: tuple[Metric, ...] = default_metrics() if metrics is None else tuple(metrics)

    def run_suite(self, suite: EvalSuite, *, tags: Collection[str] = ()) -> SuiteResult:
        """Run every case in ``suite``, or only those carrying one of ``tags``."""
        started_at = datetime.now(UTC).isoformat(timespec="seconds")
        start = time.perf_counter()
        results = tuple(
            self.run_case(
                case,
                suite_timeout_seconds=suite.default_timeout_seconds,
                pricing=suite.pricing,
            )
            for case in suite.select(tags)
        )
        return SuiteResult(
            suite_name=suite.name,
            agent_name=self.adapter.name,
            started_at=started_at,
            duration_seconds=time.perf_counter() - start,
            cases=results,
            source=suite.source,
        )

    def run_case(
        self,
        case: EvalCase,
        *,
        suite_timeout_seconds: float | None = None,
        pricing: Pricing | None = None,
    ) -> CaseResult:
        timeout = case.timeout_seconds or suite_timeout_seconds or self.default_timeout_seconds
        call = _invoke(self.adapter, AgentRequest(case.input, case.context), timeout)

        def result(status: CaseStatus, error: str | None = None) -> CaseResult:
            return CaseResult(
                case_id=case.id,
                status=status,
                latency_seconds=call.latency_seconds,
                timeout_seconds=timeout,
                description=case.description,
                tags=case.tags,
                error=error,
            )

        if call.timed_out:
            return result(CaseStatus.TIMEOUT, f"no response within the {timeout:g}s timeout")
        if call.error is not None:
            return result(CaseStatus.ERROR, f"{type(call.error).__name__}: {call.error}")
        response = call.response
        if not isinstance(response, AgentResponse):
            return result(
                CaseStatus.ERROR,
                f"adapter returned {type(response).__name__}, expected AgentResponse",
            )

        context = MetricContext(case, response, call.latency_seconds, pricing)
        metric_results = tuple(
            outcome for metric in self.metrics if (outcome := _score(metric, context)) is not None
        )
        failed = any(outcome.passed is False for outcome in metric_results)
        return CaseResult(
            case_id=case.id,
            status=CaseStatus.FAILED if failed else CaseStatus.PASSED,
            latency_seconds=call.latency_seconds,
            timeout_seconds=timeout,
            description=case.description,
            tags=case.tags,
            output=response.output,
            tool_calls=tuple(response.tool_calls),
            usage=response.usage,
            cost_usd=resolve_cost(response.usage, pricing)[0],
            metrics=metric_results,
        )


def _score(metric: Metric, context: MetricContext) -> MetricResult | None:
    """Run one metric; a metric that crashes fails the case instead of the run."""
    try:
        return metric.evaluate(context)
    except Exception as exc:
        return MetricResult(metric.name, False, None, f"metric raised {type(exc).__name__}: {exc}")


@dataclass(slots=True)
class _Call:
    response: object = None
    error: Exception | None = None
    timed_out: bool = False
    latency_seconds: float = 0.0


async def _await_with_deadline(awaitable: object, timeout: float) -> tuple[object, bool]:
    """Await an adapter's coroutine; return ``(result, deadline_expired)``."""
    scope = asyncio.timeout(timeout)
    try:
        async with scope:
            return await awaitable, False  # type: ignore[misc]
    except TimeoutError:
        if scope.expired():
            return None, True
        raise  # the adapter's own TimeoutError: report it as an error


def _invoke(adapter: AgentAdapter, request: AgentRequest, timeout: float) -> _Call:
    call = _Call()

    def target() -> None:
        start = time.perf_counter()
        try:
            outcome: object = adapter.run(request)
            if inspect.isawaitable(outcome):
                outcome, call.timed_out = asyncio.run(_await_with_deadline(outcome, timeout))
            call.response = outcome
        except Exception as exc:
            call.error = exc
        finally:
            call.latency_seconds = time.perf_counter() - start

    start = time.perf_counter()
    thread = threading.Thread(target=target, name=f"eval-{adapter.name}", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        # Abandon the worker. Return a fresh record: the worker may still
        # write to `call` when it eventually finishes.
        return _Call(timed_out=True, latency_seconds=time.perf_counter() - start)
    return call
