"""Metrics that score one agent response against one evaluation case.

A metric is any object with a ``name`` and an ``evaluate`` method (see
:class:`Metric`). The four built-ins are deterministic and need no model or
network access.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from david_agent_evals.cases import (
    ArgumentMatchMode,
    EvalCase,
    ExpectedToolCall,
    Pricing,
    ToolMatchMode,
)
from david_agent_evals.regex import DEFAULT_REGEX_TIMEOUT_SECONDS, bounded_search
from david_agent_evals.types import AgentResponse, ToolCall, Usage


@dataclass(frozen=True, slots=True)
class MetricContext:
    """Everything a metric may look at for one case."""

    case: EvalCase
    response: AgentResponse
    latency_seconds: float
    pricing: Pricing | None = None


@dataclass(frozen=True, slots=True)
class MetricResult:
    name: str
    passed: bool | None
    """``True``/``False`` when the case set a requirement for this metric;
    ``None`` when the case set none and the metric is informational. A
    requirement that cannot be checked is ``False``, never ``None``."""
    value: float | None
    """The measurement: a 0-1 score, seconds, or USD depending on the metric."""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": self.value,
            "detail": self.detail,
        }


class Metric(Protocol):
    """Interface for custom metrics. Return ``None`` when not applicable."""

    @property
    def name(self) -> str: ...

    def evaluate(self, context: MetricContext) -> MetricResult | None: ...


class TaskSuccess:
    """Did the agent's final output satisfy the case's ``expect.output`` checks?

    A ``matches`` pattern is searched under a hard time limit (see
    :mod:`david_agent_evals.regex`). A search that cannot be completed in time
    fails the check: an unanswered requirement is not a pass.
    """

    name = "task_success"

    def __init__(self, *, regex_timeout_seconds: float = DEFAULT_REGEX_TIMEOUT_SECONDS) -> None:
        if regex_timeout_seconds <= 0:
            raise ValueError("regex_timeout_seconds must be greater than 0")
        self.regex_timeout_seconds = regex_timeout_seconds

    def evaluate(self, context: MetricContext) -> MetricResult | None:
        expected = context.case.expect.output
        if expected is None:
            return None
        output = context.response.output
        fold: Callable[[str], str] = str.casefold if expected.ignore_case else str
        haystack = fold(output)
        failures: list[str] = []
        checks = 0

        if expected.equals is not None:
            checks += 1
            if fold(output.strip()) != fold(expected.equals.strip()):
                failures.append(f"output does not equal {expected.equals!r}")
        for needle in expected.contains:
            checks += 1
            if fold(needle) not in haystack:
                failures.append(f"output does not contain {needle!r}")
        for needle in expected.not_contains:
            checks += 1
            if fold(needle) in haystack:
                failures.append(f"output contains forbidden text {needle!r}")
        if expected.matches is not None:
            checks += 1
            outcome = bounded_search(
                expected.matches,
                output,
                ignore_case=expected.ignore_case,
                timeout_seconds=self.regex_timeout_seconds,
            )
            if outcome.matched is None:
                failures.append(f"could not check /{expected.matches}/: {outcome.problem}")
            elif not outcome.matched:
                failures.append(f"output does not match /{expected.matches}/")

        passed = not failures
        detail = f"all {checks} output check(s) passed" if passed else "; ".join(failures)
        return MetricResult(self.name, passed, 1.0 if passed else 0.0, detail)


class ToolCallCorrectness:
    """Did the agent call the expected tools, with the expected arguments?

    The value is the fraction of expected calls that were matched (for
    ``exact`` mode, relative to the longer of the expected and actual
    sequences, so extra calls lower it). Calling a forbidden tool fails the
    metric and sets the value to 0.
    """

    name = "tool_call_correctness"

    def evaluate(self, context: MetricContext) -> MetricResult | None:
        expected = context.case.expect.tool_calls
        if expected is None:
            return None
        actual = list(context.response.tool_calls)
        called = ", ".join(call.name for call in actual) or "no tools"

        violations = sorted({call.name for call in actual if call.name in expected.forbidden})
        if violations:
            detail = f"called forbidden tool(s): {', '.join(violations)}; agent called: {called}"
            return MetricResult(self.name, False, 0.0, detail)
        if expected.calls is None:
            return MetricResult(self.name, True, 1.0, "no forbidden tools were called")

        def same(want: ExpectedToolCall, got: ToolCall) -> bool:
            return want.name == got.name and (
                want.arguments is None
                or _values_match(want.arguments, got.arguments, expected.arguments_mode)
            )

        wanted = expected.calls
        if expected.mode is ToolMatchMode.EXACT:
            matched = sum(same(w, g) for w, g in zip(wanted, actual, strict=False))
            denominator = max(len(wanted), len(actual))
            passed = matched == len(wanted) == len(actual)
        else:
            count = _count_in_order if expected.mode is ToolMatchMode.IN_ORDER else _count_any_order
            matched = count(wanted, actual, same)
            denominator = len(wanted)
            passed = matched == len(wanted)

        score = matched / denominator if denominator else 1.0
        detail = (
            f"matched {matched} of {len(wanted)} expected call(s) "
            f"({expected.mode.value}); agent called: {called}"
        )
        return MetricResult(self.name, passed, score, detail)


class Latency:
    """Wall-clock seconds the adapter took. Enforced only if the case sets
    ``expect.max_latency_seconds``; otherwise informational."""

    name = "latency"

    def evaluate(self, context: MetricContext) -> MetricResult | None:
        limit = context.case.expect.max_latency_seconds
        seconds = context.latency_seconds
        if limit is None:
            return MetricResult(self.name, None, seconds, "no latency limit set")
        passed = seconds <= limit
        comparison = "within" if passed else "exceeds"
        return MetricResult(
            self.name, passed, seconds, f"{seconds:.4f}s {comparison} limit of {limit:g}s"
        )


class Cost:
    """Cost in USD, when the agent reported enough usage data to know it.

    The framework never estimates a cost it was not given the data for, so the
    value is ``None`` when no cost can be determined. If the case also sets
    ``expect.max_cost_usd``, the metric then fails: a limit that cannot be
    checked has not been met, and reporting it as anything but a failure would
    let a case with a cost requirement pass unverified. To evaluate such a
    case, have the adapter report usage (and give the suite ``pricing`` if it
    reports only token counts), or remove the limit.
    """

    name = "cost"

    def evaluate(self, context: MetricContext) -> MetricResult | None:
        cost, source = resolve_cost(context.response.usage, context.pricing)
        limit = context.case.expect.max_cost_usd
        if cost is None:
            if limit is None:
                return MetricResult(self.name, None, None, f"cost unavailable: {source}")
            detail = f"max_cost_usd of ${limit:g} cannot be checked: cost unavailable ({source})"
            return MetricResult(self.name, False, None, detail)
        if limit is None:
            return MetricResult(self.name, None, cost, source)
        passed = cost <= limit
        comparison = "within" if passed else "exceeds"
        return MetricResult(
            self.name, passed, cost, f"${cost:.6f} {comparison} limit of ${limit:g} ({source})"
        )


def default_metrics() -> tuple[Metric, ...]:
    return (TaskSuccess(), ToolCallCorrectness(), Latency(), Cost())


def resolve_cost(usage: Usage | None, pricing: Pricing | None) -> tuple[float | None, str]:
    """Return ``(cost_usd, explanation)``; cost is ``None`` when unknowable."""
    if usage is None:
        return None, "agent reported no usage data"
    if usage.cost_usd is not None:
        return usage.cost_usd, "cost reported by adapter"
    if usage.input_tokens is None or usage.output_tokens is None:
        return None, "agent reported neither a cost nor complete token counts"
    if pricing is None:
        return None, "token counts reported but the suite defines no pricing"
    cost = pricing.cost_usd(usage.input_tokens, usage.output_tokens)
    return cost, "computed from reported token counts and suite pricing"


def _values_match(expected: Any, actual: Any, mode: ArgumentMatchMode) -> bool:
    """Compare argument values. Mappings follow ``mode`` recursively; every
    other value (including lists) must be equal."""
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if mode is ArgumentMatchMode.EXACT and set(expected) != set(actual):
            return False
        return all(
            key in actual and _values_match(value, actual[key], mode)
            for key, value in expected.items()
        )
    if isinstance(expected, bool) != isinstance(actual, bool):
        return False  # keep True from matching 1
    return bool(expected == actual)


_Same = Callable[[ExpectedToolCall, ToolCall], bool]


def _count_in_order(
    wanted: Sequence[ExpectedToolCall], actual: Sequence[ToolCall], same: _Same
) -> int:
    """Longest run of expected calls found, in order, within the actual calls."""
    previous = [0] * (len(actual) + 1)
    for want in wanted:
        current = [0]
        for j, got in enumerate(actual, start=1):
            if same(want, got):
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def _count_any_order(
    wanted: Sequence[ExpectedToolCall], actual: Sequence[ToolCall], same: _Same
) -> int:
    """Most expected calls that can each be paired with a distinct actual call."""
    owner: dict[int, int] = {}  # actual index -> expected index

    def assign(i: int, visited: set[int]) -> bool:
        for j, got in enumerate(actual):
            if j in visited or not same(wanted[i], got):
                continue
            visited.add(j)
            if j not in owner or assign(owner[j], visited):
                owner[j] = i
                return True
        return False

    return sum(assign(i, set()) for i in range(len(wanted)))
