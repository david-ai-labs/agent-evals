from __future__ import annotations

from typing import Any

import pytest

from david_agent_evals import AgentResponse, MetricContext, Pricing, ToolCall, Usage
from david_agent_evals.metrics import (
    Cost,
    Latency,
    MetricResult,
    TaskSuccess,
    ToolCallCorrectness,
    resolve_cost,
)

from .conftest import make_case


def score_output(expect: dict[str, Any], output: str) -> MetricResult:
    context = MetricContext(make_case({"output": expect}), AgentResponse(output), 0.0)
    result = TaskSuccess().evaluate(context)
    assert result is not None
    return result


def score_tools(expect: dict[str, Any], *calls: ToolCall) -> MetricResult:
    case = make_case({"tool_calls": expect})
    result = ToolCallCorrectness().evaluate(
        MetricContext(case, AgentResponse("", tool_calls=calls), 0.0)
    )
    assert result is not None
    return result


LOOKUP = ToolCall("lookup", {"id": "A-1", "verbose": True})
REFUND = ToolCall("refund", {"id": "A-1", "amount": 10})
NOTE = ToolCall("add_note", {"text": "done"})


class TestTaskSuccess:
    def test_not_applicable_without_output_expectation(self) -> None:
        context = MetricContext(make_case(), AgentResponse("anything"), 0.0)
        assert TaskSuccess().evaluate(context) is None

    @pytest.mark.parametrize(
        ("expect", "output", "passed"),
        [
            ({"equals": "Paris"}, "  Paris\n", True),
            ({"equals": "Paris"}, "Paris, France", False),
            ({"equals": "paris", "ignore_case": True}, "PARIS", True),
            ({"contains": ["refund", "A-1"]}, "refund issued for A-1", True),
            ({"contains": ["refund", "A-2"]}, "refund issued for A-1", False),
            ({"contains": "Refund"}, "refund issued", False),
            ({"contains": "Refund", "ignore_case": True}, "refund issued", True),
            ({"not_contains": "password"}, "all good", True),
            ({"not_contains": "password"}, "your password is hunter2", False),
            ({"matches": r"order \w-\d+"}, "your order A-1 shipped", True),
            ({"matches": r"^\d+$"}, "abc", False),
            ({"matches": "SHIPPED", "ignore_case": True}, "it shipped", True),
        ],
    )
    def test_output_checks(self, expect: dict[str, Any], output: str, passed: bool) -> None:
        result = score_output(expect, output)
        assert result.passed is passed
        assert result.value == (1.0 if passed else 0.0)

    def test_every_failed_check_is_explained(self) -> None:
        result = score_output({"contains": ["alpha", "beta"], "not_contains": "gamma"}, "gamma")
        assert result.passed is False
        assert "'alpha'" in result.detail
        assert "'beta'" in result.detail
        assert "forbidden text 'gamma'" in result.detail


class TestToolCallCorrectness:
    def test_not_applicable_without_tool_expectation(self) -> None:
        context = MetricContext(make_case(), AgentResponse("", tool_calls=[LOOKUP]), 0.0)
        assert ToolCallCorrectness().evaluate(context) is None

    def test_exact_match_passes(self) -> None:
        result = score_tools({"calls": [{"name": "lookup"}, {"name": "refund"}]}, LOOKUP, REFUND)
        assert (result.passed, result.value) == (True, 1.0)

    def test_exact_rejects_wrong_order(self) -> None:
        result = score_tools({"calls": [{"name": "lookup"}, {"name": "refund"}]}, REFUND, LOOKUP)
        assert (result.passed, result.value) == (False, 0.0)

    def test_exact_rejects_extra_calls_and_lowers_score(self) -> None:
        result = score_tools({"calls": [{"name": "lookup"}]}, LOOKUP, NOTE)
        assert result.passed is False
        assert result.value == pytest.approx(0.5)

    def test_exact_rejects_missing_calls(self) -> None:
        result = score_tools({"calls": [{"name": "lookup"}, {"name": "refund"}]}, LOOKUP)
        assert result.passed is False
        assert result.value == pytest.approx(0.5)
        assert "matched 1 of 2" in result.detail

    def test_exact_empty_expectation_requires_no_calls(self) -> None:
        assert score_tools({"calls": []}).passed is True
        failed = score_tools({"calls": []}, LOOKUP)
        assert (failed.passed, failed.value) == (False, 0.0)

    def test_in_order_allows_interleaved_calls(self) -> None:
        expect = {"match": "in_order", "calls": [{"name": "lookup"}, {"name": "refund"}]}
        assert score_tools(expect, LOOKUP, NOTE, REFUND).passed is True
        wrong_order = score_tools(expect, REFUND, NOTE, LOOKUP)
        assert wrong_order.passed is False
        assert wrong_order.value == pytest.approx(0.5)

    def test_any_order_ignores_order_and_extras(self) -> None:
        expect = {"match": "any_order", "calls": [{"name": "lookup"}, {"name": "refund"}]}
        assert score_tools(expect, REFUND, NOTE, LOOKUP).passed is True
        assert score_tools(expect, REFUND, NOTE).value == pytest.approx(0.5)

    def test_any_order_pairs_each_expected_call_with_a_distinct_actual_call(self) -> None:
        # Two expected lookups cannot both be satisfied by a single actual one.
        expect = {"match": "any_order", "calls": [{"name": "lookup"}, {"name": "lookup"}]}
        assert score_tools(expect, LOOKUP).passed is False
        assert score_tools(expect, LOOKUP, LOOKUP).passed is True

    def test_any_order_finds_a_pairing_a_greedy_match_would_miss(self) -> None:
        # The unconstrained expectation must yield the id=1 call to the specific one.
        expect = {
            "match": "any_order",
            "calls": [{"name": "get"}, {"name": "get", "arguments": {"id": 1}}],
        }
        result = score_tools(expect, ToolCall("get", {"id": 1}), ToolCall("get", {"id": 2}))
        assert result.passed is True

    def test_subset_arguments_allow_extras_but_not_mismatches(self) -> None:
        assert score_tools(
            {"calls": [{"name": "lookup", "arguments": {"id": "A-1"}}]}, LOOKUP
        ).passed
        wrong = score_tools({"calls": [{"name": "lookup", "arguments": {"id": "A-2"}}]}, LOOKUP)
        assert wrong.passed is False
        missing = score_tools({"calls": [{"name": "lookup", "arguments": {"page": 1}}]}, LOOKUP)
        assert missing.passed is False

    def test_exact_arguments_reject_extras(self) -> None:
        expect = {
            "arguments_match": "exact",
            "calls": [{"name": "lookup", "arguments": {"id": "A-1"}}],
        }
        assert score_tools(expect, LOOKUP).passed is False
        assert score_tools(expect, ToolCall("lookup", {"id": "A-1"})).passed is True

    def test_nested_arguments_follow_the_match_mode(self) -> None:
        call = ToolCall("search", {"filter": {"status": "open", "limit": 5}, "ids": [1, 2]})
        subset = {"calls": [{"name": "search", "arguments": {"filter": {"status": "open"}}}]}
        assert score_tools(subset, call).passed is True
        exact = {**subset, "arguments_match": "exact"}
        assert score_tools(exact, call).passed is False
        lists = {"calls": [{"name": "search", "arguments": {"ids": [1]}}]}
        assert score_tools(lists, call).passed is False

    def test_booleans_do_not_match_integers(self) -> None:
        expect = {"calls": [{"name": "lookup", "arguments": {"verbose": 1}}]}
        assert score_tools(expect, LOOKUP).passed is False

    def test_forbidden_tool_fails_and_zeroes_the_score(self) -> None:
        expect = {"calls": [{"name": "lookup"}, {"name": "refund"}], "forbidden": ["refund"]}
        result = score_tools(expect, LOOKUP, REFUND)
        assert (result.passed, result.value) == (False, 0.0)
        assert "forbidden tool(s): refund" in result.detail

    def test_forbidden_only_expectation(self) -> None:
        assert score_tools({"forbidden": ["refund"]}, LOOKUP, NOTE).passed is True
        assert score_tools({"forbidden": ["refund"]}).passed is True
        assert score_tools({"forbidden": ["refund"]}, REFUND).passed is False


class TestLatency:
    def test_informational_without_a_limit(self) -> None:
        result = Latency().evaluate(MetricContext(make_case(), AgentResponse(""), 0.42))
        assert result is not None
        assert (result.passed, result.value) == (None, 0.42)

    @pytest.mark.parametrize(("seconds", "passed"), [(0.5, True), (1.0, True), (1.01, False)])
    def test_limit_is_enforced(self, seconds: float, passed: bool) -> None:
        case = make_case({"max_latency_seconds": 1.0})
        result = Latency().evaluate(MetricContext(case, AgentResponse(""), seconds))
        assert result is not None
        assert (result.passed, result.value) == (passed, seconds)


class TestCost:
    PRICING = Pricing(input_per_million_usd=2.0, output_per_million_usd=10.0)

    def test_reported_cost_wins_over_pricing(self) -> None:
        usage = Usage(input_tokens=1000, output_tokens=1000, cost_usd=0.5)
        assert resolve_cost(usage, self.PRICING) == (0.5, "cost reported by adapter")

    def test_cost_is_computed_from_tokens_and_pricing(self) -> None:
        cost, source = resolve_cost(Usage(input_tokens=1000, output_tokens=500), self.PRICING)
        assert cost == pytest.approx(0.002 + 0.005)
        assert "suite pricing" in source

    @pytest.mark.parametrize(
        ("usage", "pricing"),
        [
            (None, PRICING),
            (Usage(), PRICING),
            (Usage(input_tokens=10), PRICING),
            (Usage(input_tokens=10, output_tokens=10), None),
        ],
    )
    def test_cost_is_unavailable_rather_than_guessed(
        self, usage: Usage | None, pricing: Pricing | None
    ) -> None:
        cost, _ = resolve_cost(usage, pricing)
        assert cost is None

    def test_zero_cost_is_a_real_value(self) -> None:
        assert resolve_cost(Usage(cost_usd=0.0), None)[0] == 0.0

    def test_budget_is_enforced_when_cost_is_known(self) -> None:
        case = make_case({"max_cost_usd": 0.01})
        within = Cost().evaluate(
            MetricContext(case, AgentResponse("", usage=Usage(cost_usd=0.01)), 0)
        )
        over = Cost().evaluate(
            MetricContext(case, AgentResponse("", usage=Usage(cost_usd=0.02)), 0)
        )
        assert within is not None
        assert over is not None
        assert (within.passed, over.passed) == (True, False)

    def test_budget_is_not_evaluated_when_cost_is_unknown(self) -> None:
        case = make_case({"max_cost_usd": 0.01})
        result = Cost().evaluate(MetricContext(case, AgentResponse(""), 0))
        assert result is not None
        assert (result.passed, result.value) == (None, None)
        assert "max_cost_usd was not evaluated" in result.detail

    def test_informational_without_a_budget(self) -> None:
        context = MetricContext(make_case(), AgentResponse("", usage=Usage(cost_usd=0.3)), 0)
        result = Cost().evaluate(context)
        assert result is not None
        assert (result.passed, result.value) == (None, 0.3)
