from __future__ import annotations

import asyncio
import threading
import time

import pytest

from david_agent_evals import (
    AgentAdapter,
    AgentRequest,
    AgentResponse,
    CaseStatus,
    EvalRunner,
    FunctionAdapter,
    MetricContext,
    MetricResult,
    ToolCall,
    Usage,
)
from david_agent_evals.metrics import TaskSuccess
from david_agent_evals.mock import EchoAgent, ScriptedAgent, ScriptedTurn, UnscriptedInputError
from david_agent_evals.report import render_text
from david_agent_evals.results import percentile, summarize

from .conftest import make_case, make_suite


def test_passing_case_records_the_response() -> None:
    response = AgentResponse(
        "hi there", tool_calls=[ToolCall("greet", {"who": "you"})], usage=Usage(cost_usd=0.25)
    )
    agent = ScriptedAgent("scripted", {"hello": response})
    case = make_case({"output": {"contains": "hi"}, "tool_calls": {"calls": [{"name": "greet"}]}})

    result = EvalRunner(agent).run_case(case)

    assert result.status is CaseStatus.PASSED
    assert result.output == "hi there"
    assert result.tool_calls == (ToolCall("greet", {"who": "you"}),)
    assert result.cost_usd == 0.25
    assert result.error is None
    assert [metric.name for metric in result.metrics] == [
        "task_success",
        "tool_call_correctness",
        "latency",
        "cost",
    ]


def test_failing_metric_fails_the_case() -> None:
    result = EvalRunner(EchoAgent()).run_case(make_case({"output": {"equals": "goodbye"}}))
    assert result.status is CaseStatus.FAILED
    assert result.output == "hello"


def test_case_without_expectations_passes_if_the_agent_responds() -> None:
    result = EvalRunner(EchoAgent()).run_case(make_case())
    assert result.status is CaseStatus.PASSED
    # Only the informational metrics apply.
    assert [metric.name for metric in result.metrics] == ["latency", "cost"]
    assert all(metric.passed is None for metric in result.metrics)


def test_request_carries_input_and_context() -> None:
    seen: list[AgentRequest] = []

    def record(request: AgentRequest) -> AgentResponse:
        seen.append(request)
        return AgentResponse("ok")

    case = make_case(input=[{"role": "user", "content": "hi"}], context={"locale": "de"})
    EvalRunner(FunctionAdapter("recorder", record)).run_case(case)

    assert seen == [AgentRequest([{"role": "user", "content": "hi"}], {"locale": "de"})]


def test_adapter_exception_becomes_an_error_result() -> None:
    agent = ScriptedAgent("broken", {"hello": ScriptedTurn(raises=RuntimeError("backend down"))})
    result = EvalRunner(agent).run_case(make_case({"output": {"contains": "x"}}))

    assert result.status is CaseStatus.ERROR
    assert result.error == "RuntimeError: backend down"
    assert result.metrics == ()
    assert result.output is None


def test_scripted_agent_rejects_unknown_input() -> None:
    result = EvalRunner(ScriptedAgent("scripted", {})).run_case(make_case())
    assert result.status is CaseStatus.ERROR
    assert result.error is not None
    assert result.error.startswith(UnscriptedInputError.__name__)


def test_invalid_return_type_becomes_an_error_result() -> None:
    adapter = FunctionAdapter("bad", lambda request: "just a string")  # type: ignore[arg-type,return-value]
    result = EvalRunner(adapter).run_case(make_case())
    assert result.status is CaseStatus.ERROR
    assert result.error == "adapter returned str, expected AgentResponse"


def test_blocking_adapter_times_out() -> None:
    release = threading.Event()

    def block(request: AgentRequest) -> AgentResponse:
        release.wait(10)
        return AgentResponse("too late")

    try:
        start = time.perf_counter()
        result = EvalRunner(FunctionAdapter("blocker", block)).run_case(
            make_case(timeout_seconds=0.05)
        )
        elapsed = time.perf_counter() - start
    finally:
        release.set()

    assert result.status is CaseStatus.TIMEOUT
    assert result.error == "no response within the 0.05s timeout"
    assert result.timeout_seconds == 0.05
    assert result.output is None
    assert 0.05 <= result.latency_seconds < 5
    assert elapsed < 5, "the runner must not wait for the blocked adapter"


def test_async_adapter_is_awaited() -> None:
    async def respond(request: AgentRequest) -> AgentResponse:
        await asyncio.sleep(0)
        return AgentResponse(f"async:{request.input}")

    result = EvalRunner(FunctionAdapter("async", respond)).run_case(
        make_case({"output": {"equals": "async:hello"}})
    )
    assert result.status is CaseStatus.PASSED


def test_async_adapter_times_out_and_is_cancelled() -> None:
    cancelled = threading.Event()

    async def hang(request: AgentRequest) -> AgentResponse:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return AgentResponse("too late")

    result = EvalRunner(FunctionAdapter("hanger", hang)).run_case(make_case(timeout_seconds=0.05))

    assert result.status is CaseStatus.TIMEOUT
    assert cancelled.wait(2), "the timed-out coroutine should have been cancelled"


def test_adapters_own_timeout_error_is_an_error_not_a_runner_timeout() -> None:
    async def upstream_timeout(request: AgentRequest) -> AgentResponse:
        raise TimeoutError("upstream gave up")

    result = EvalRunner(FunctionAdapter("upstream", upstream_timeout)).run_case(make_case())
    assert result.status is CaseStatus.ERROR
    assert result.error == "TimeoutError: upstream gave up"


def test_runner_works_inside_a_running_event_loop() -> None:
    async def respond(request: AgentRequest) -> AgentResponse:
        return AgentResponse("ok")

    async def main() -> CaseStatus:
        return EvalRunner(FunctionAdapter("async", respond)).run_case(make_case()).status

    assert asyncio.run(main()) is CaseStatus.PASSED


def test_timeout_precedence_case_then_suite_then_runner() -> None:
    suite = make_suite(
        {"id": "own", "input": "x", "timeout_seconds": 1},
        {"id": "inherits", "input": "x"},
        defaults={"timeout_seconds": 2},
    )
    bare_suite = make_suite({"id": "bare", "input": "x"})
    runner = EvalRunner(EchoAgent(), default_timeout_seconds=3)

    timeouts = {r.case_id: r.timeout_seconds for r in runner.run_suite(suite).cases}
    timeouts |= {r.case_id: r.timeout_seconds for r in runner.run_suite(bare_suite).cases}

    assert timeouts == {"own": 1, "inherits": 2, "bare": 3}


def test_runner_rejects_non_positive_default_timeout() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        EvalRunner(EchoAgent(), default_timeout_seconds=0)


def test_run_suite_reports_each_status_and_uses_suite_pricing() -> None:
    agent = ScriptedAgent(
        "mixed",
        {
            "pass": AgentResponse("yes", usage=Usage(input_tokens=1_000_000, output_tokens=0)),
            "fail": AgentResponse("no", usage=Usage(cost_usd=0.5)),
            "boom": ScriptedTurn(raises=ValueError("bad input")),
            "slow": ScriptedTurn(response=AgentResponse("yes"), delay_seconds=1.0),
        },
    )
    expect = {"output": {"equals": "yes"}}
    suite = make_suite(
        {"id": "a", "input": "pass", "expect": expect},
        {"id": "b", "input": "fail", "expect": expect},
        {"id": "c", "input": "boom", "expect": expect},
        {"id": "d", "input": "slow", "expect": expect, "timeout_seconds": 0.05},
        pricing={"input_per_million_usd": 3, "output_per_million_usd": 15},
    )

    result = EvalRunner(agent).run_suite(suite)

    assert result.suite_name == "test-suite"
    assert result.agent_name == "mixed"
    assert [case.status for case in result.cases] == [
        CaseStatus.PASSED,
        CaseStatus.FAILED,
        CaseStatus.ERROR,
        CaseStatus.TIMEOUT,
    ]
    assert result.all_passed is False

    summary = result.summary
    assert (summary.total, summary.passed, summary.failed) == (4, 1, 1)
    assert (summary.errors, summary.timeouts) == (1, 1)
    assert summary.pass_rate == 0.25
    assert summary.total_cost_usd == pytest.approx(3.0 + 0.5)
    assert summary.cases_with_cost == 2
    assert summary.latency is not None
    assert summary.latency.count == 2, "only responding cases count towards latency"
    task_success = next(m for m in summary.metrics if m.name == "task_success")
    assert (task_success.evaluated, task_success.passed) == (2, 1)
    assert task_success.mean_value == 0.5


def test_cost_limit_without_cost_data_fails_the_case() -> None:
    # Regression: this case used to pass because the limit was skipped.
    expect = {"output": {"equals": "ok"}, "max_cost_usd": 0.01}
    suite = make_suite(
        {"id": "no-usage", "input": "none", "expect": expect},
        {"id": "tokens-no-pricing", "input": "tokens", "expect": expect},
        {"id": "cost-reported", "input": "cost", "expect": expect},
    )
    agent = ScriptedAgent(
        "s",
        {
            "none": AgentResponse("ok"),
            "tokens": AgentResponse("ok", usage=Usage(input_tokens=5, output_tokens=5)),
            "cost": AgentResponse("ok", usage=Usage(cost_usd=0.001)),
        },
    )

    result = EvalRunner(agent).run_suite(suite)

    assert [case.status for case in result.cases] == [
        CaseStatus.FAILED,
        CaseStatus.FAILED,
        CaseStatus.PASSED,
    ]
    cost = next(m for m in result.summary.metrics if m.name == "cost")
    assert (cost.evaluated, cost.passed) == (3, 1)
    assert "max_cost_usd of $0.01 cannot be checked" in render_text([result])


def test_catastrophic_regex_fails_its_case_and_the_run_continues() -> None:
    suite = make_suite(
        {"id": "hostile", "input": "a" * 64 + "b", "expect": {"output": {"matches": "(a+)+$"}}},
        {"id": "next", "input": "fine", "expect": {"output": {"matches": "^fine$"}}},
    )
    runner = EvalRunner(EchoAgent(), metrics=[TaskSuccess(regex_timeout_seconds=0.3)])

    start = time.perf_counter()
    result = runner.run_suite(suite)

    assert time.perf_counter() - start < 10
    assert [case.status for case in result.cases] == [CaseStatus.FAILED, CaseStatus.PASSED]
    assert "search did not finish within 0.3s" in result.cases[0].metrics[0].detail


def test_run_suite_filters_by_tag() -> None:
    suite = make_suite(
        {"id": "a", "input": "x", "tags": ["smoke"]},
        {"id": "b", "input": "x"},
    )
    result = EvalRunner(EchoAgent()).run_suite(suite, tags=["smoke"])
    assert [case.case_id for case in result.cases] == ["a"]


def test_custom_metrics_replace_the_defaults() -> None:
    class Politeness:
        name = "politeness"

        def evaluate(self, context: MetricContext) -> MetricResult | None:
            polite = "please" in context.response.output
            return MetricResult(self.name, polite, float(polite))

    runner = EvalRunner(EchoAgent(), metrics=[Politeness()])
    assert runner.run_case(make_case(input="please help")).status is CaseStatus.PASSED
    rude = runner.run_case(make_case(input="help"))
    assert rude.status is CaseStatus.FAILED
    assert [metric.name for metric in rude.metrics] == ["politeness"]


def test_crashing_metric_fails_the_case_without_stopping_the_run() -> None:
    class Broken:
        name = "broken"

        def evaluate(self, context: MetricContext) -> MetricResult | None:
            raise ZeroDivisionError("oops")

    suite = make_suite({"id": "a", "input": "x"}, {"id": "b", "input": "y"})
    result = EvalRunner(EchoAgent(), metrics=[Broken()]).run_suite(suite)

    assert [case.status for case in result.cases] == [CaseStatus.FAILED, CaseStatus.FAILED]
    assert result.cases[0].metrics[0].detail == "metric raised ZeroDivisionError: oops"


def test_mock_agents_satisfy_the_adapter_protocol() -> None:
    assert isinstance(EchoAgent(), AgentAdapter)
    assert isinstance(ScriptedAgent("s", {}), AgentAdapter)
    assert isinstance(FunctionAdapter("f", lambda request: AgentResponse("")), AgentAdapter)
    assert not isinstance(object(), AgentAdapter)


def test_scripted_turn_needs_exactly_one_outcome() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ScriptedTurn()
    with pytest.raises(ValueError, match="exactly one"):
        ScriptedTurn(response=AgentResponse(""), raises=RuntimeError())


def test_scripted_agent_default_turn() -> None:
    agent = ScriptedAgent("s", {}, default=AgentResponse("fallback"))
    assert agent.run(AgentRequest("anything")).output == "fallback"


@pytest.mark.parametrize(
    ("percent", "expected"),
    [(50, 5.0), (95, 10.0), (100, 10.0), (10, 1.0), (1, 1.0)],
)
def test_percentile_is_nearest_rank(percent: float, expected: float) -> None:
    values = [float(n) for n in range(1, 11)]
    assert percentile(values, percent) == expected


def test_percentile_of_single_value_and_empty_input() -> None:
    assert percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)


def test_summary_of_no_cases() -> None:
    summary = summarize([])
    assert (summary.total, summary.pass_rate) == (0, 0.0)
    assert summary.latency is None
    assert summary.total_cost_usd is None
