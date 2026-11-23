"""End-to-end tests: the agent loader, the CLI, reports, and the shipped examples."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

import david_agent_evals
from david_agent_evals import AgentResponse, CaseStatus, EvalRunner, ToolCall, load_suite
from david_agent_evals.cli import main
from david_agent_evals.loader import AgentLoadError, load_agent
from david_agent_evals.mock import ScriptedAgent
from david_agent_evals.report import render_text, to_json

from .conftest import EXAMPLES, make_suite

SUPPORT = str(EXAMPLES / "cases" / "support.yaml")
POLICY = str(EXAMPLES / "cases" / "support_policy.json")
REFERENCE = f"{EXAMPLES / 'agents.py'}:reference_agent"
FAULTY = f"{EXAMPLES / 'agents.py'}:faulty_agent"


# --- shipped examples -------------------------------------------------------


@pytest.mark.parametrize("suite_path", [SUPPORT, POLICY])
def test_reference_agent_passes_every_example_case(suite_path: str) -> None:
    result = EvalRunner(load_agent(REFERENCE)).run_suite(load_suite(suite_path))
    failures = {case.case_id: case.status for case in result.cases if not result.all_passed}
    assert failures == {}
    assert len(result.cases) == len(load_suite(suite_path).cases)


def test_faulty_agent_fails_in_the_documented_ways() -> None:
    agent = load_agent(FAULTY)
    support = EvalRunner(agent).run_suite(load_suite(SUPPORT))
    policy = EvalRunner(agent).run_suite(load_suite(POLICY))

    assert {case.case_id: case.status.value for case in support.cases} == {
        "refund-eligible-order": "failed",
        "refund-ineligible-order": "failed",
        "order-status": "timeout",
        "unknown-order": "passed",
        "refund-missing-order-number": "passed",
        "escalation-request": "error",
    }
    assert {case.case_id: case.status.value for case in policy.cases} == {
        "instruction-override-refund": "failed",
        "refund-over-limit": "failed",
        "out-of-scope-question": "passed",
    }


def test_example_cost_comes_from_simulated_tokens_and_suite_pricing() -> None:
    result = EvalRunner(load_agent(REFERENCE)).run_suite(load_suite(SUPPORT))
    case = next(c for c in result.cases if c.case_id == "refund-missing-order-number")
    assert case.usage is not None
    # "I want my money back." -> 5 words in, 10 words out, at $1 / $2 per million.
    assert (case.usage.input_tokens, case.usage.output_tokens) == (5, 10)
    assert case.cost_usd == pytest.approx((5 * 1.0 + 10 * 2.0) / 1_000_000)
    # The JSON suite defines no pricing, so no cost is reported there.
    policy = EvalRunner(load_agent(REFERENCE)).run_suite(load_suite(POLICY))
    assert policy.summary.total_cost_usd is None


# --- agent loader -----------------------------------------------------------


def test_load_agent_from_module_path() -> None:
    assert load_agent("david_agent_evals.mock:EchoAgent").name == "echo-agent"


def test_load_agent_accepts_instance_class_and_factory(tmp_path: Path) -> None:
    file = tmp_path / "my_agents.py"
    file.write_text(
        "from david_agent_evals.mock import EchoAgent\n"
        "instance = EchoAgent(name='instance')\n"
        "class Custom(EchoAgent):\n"
        "    pass\n"
        "def factory():\n"
        "    return EchoAgent(name='from-factory')\n"
        "def bad_factory():\n"
        "    return 42\n"
        "def crashing_factory():\n"
        "    raise RuntimeError('no credentials')\n"
        "not_an_agent = 42\n",
        encoding="utf-8",
    )
    assert load_agent(f"{file}:instance").name == "instance"
    assert load_agent(f"{file}:Custom").name == "echo-agent"
    assert load_agent(f"{file}:factory").name == "from-factory"
    with pytest.raises(AgentLoadError, match="not an agent adapter"):
        load_agent(f"{file}:bad_factory")
    with pytest.raises(AgentLoadError, match="RuntimeError: no credentials"):
        load_agent(f"{file}:crashing_factory")
    with pytest.raises(AgentLoadError, match="not an agent adapter"):
        load_agent(f"{file}:not_an_agent")
    with pytest.raises(AgentLoadError, match="no attribute 'missing'"):
        load_agent(f"{file}:missing")


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("no_separator", "invalid agent"),
        ("module_only:", "invalid agent"),
        (":attribute_only", "invalid agent"),
        ("definitely_not_a_module_xyz:agent", "could not import module"),
        ("missing/file.py:agent", "agent file not found"),
    ],
)
def test_load_agent_rejects_bad_specs(spec: str, message: str) -> None:
    with pytest.raises(AgentLoadError, match=message):
        load_agent(spec)


def test_load_agent_reports_import_time_failures(tmp_path: Path) -> None:
    file = tmp_path / "explodes.py"
    file.write_text("raise ValueError('bad config')\n", encoding="utf-8")
    with pytest.raises(AgentLoadError, match="ValueError: bad config"):
        load_agent(f"{file}:agent")


# --- CLI --------------------------------------------------------------------


def test_cli_run_passes_and_prints_a_report(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", SUPPORT, POLICY, "--agent", REFERENCE])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Suite: customer-support" in out
    assert "Suite: support-policy" in out
    assert "Agent: reference-support-agent" in out
    assert "6 total: 6 passed, 0 failed, 0 errors, 0 timeouts" in out
    assert "Pass rate   100.0%" in out
    assert "tool_call_correctness  6/6 passed, mean score 1.00" in out
    assert "(6 of 6 case(s) reported usage)" in out
    assert "Cost        not available" in out  # the JSON suite has no pricing


def test_cli_run_reports_failures_with_reasons(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", SUPPORT, "--agent", FAULTY])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "6 total: 2 passed, 2 failed, 1 errors, 1 timeouts" in out
    assert "called forbidden tool(s): issue_refund" in out
    assert "matched 0 of 2 expected call(s) (exact); agent called: issue_refund" in out
    assert "no response within the 0.5s timeout" in out
    assert "RuntimeError: ticketing backend unavailable" in out


def test_cli_writes_json_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "results.json"
    exit_code = main(["run", SUPPORT, "--agent", FAULTY, "--output", str(output)])

    assert exit_code == 1
    assert f"JSON results written to {output}" in capsys.readouterr().out
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "0.1"
    assert report["framework_version"] == david_agent_evals.__version__
    assert report["all_passed"] is False
    datetime.fromisoformat(report["generated_at"])

    (suite,) = report["suites"]
    assert suite["suite"] == "customer-support"
    assert suite["agent"] == "faulty-support-agent"
    assert suite["summary"]["total"] == 6
    assert suite["summary"]["pass_rate"] == pytest.approx(2 / 6)
    cases = {case["case_id"]: case for case in suite["cases"]}
    assert cases["order-status"]["status"] == "timeout"
    assert cases["order-status"]["output"] is None
    assert cases["escalation-request"]["error"] == "RuntimeError: ticketing backend unavailable"
    refund = cases["refund-eligible-order"]
    assert refund["tool_calls"] == [
        {"name": "issue_refund", "arguments": {"order_id": "A-1001", "amount_usd": 42.5}}
    ]
    assert {metric["name"] for metric in refund["metrics"]} == {
        "task_success",
        "tool_call_correctness",
        "latency",
        "cost",
    }


def test_cli_json_format_prints_only_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", POLICY, "--agent", REFERENCE, "--format", "json"])
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["all_passed"] is True
    assert [case["status"] for case in report["suites"][0]["cases"]] == ["passed"] * 3


def test_cli_tag_filter(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", SUPPORT, "--agent", REFERENCE, "--tag", "status"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "2 total: 2 passed" in out
    assert "refund-eligible-order" not in out


def test_cli_tag_matching_nothing_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", SUPPORT, "--agent", REFERENCE, "--tag", "nonexistent"])
    assert exit_code == 2
    assert "no cases match tag(s): nonexistent" in capsys.readouterr().err


def test_cli_timeout_flag_is_the_fallback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"cases": [{"id": "a", "input": "x"}]}), encoding="utf-8")
    exit_code = main(
        [
            *("run", str(suite), "--agent", "david_agent_evals.mock:EchoAgent"),
            *("--timeout", "7", "--format", "json"),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["suites"][0]["cases"][0]["timeout_seconds"] == 7


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "missing.yaml", "--agent", REFERENCE], "missing.yaml"),
        (["run", SUPPORT, "--agent", "nope_xyz:agent"], "could not import module"),
        (["validate", "missing.yaml"], "missing.yaml"),
    ],
)
def test_cli_configuration_errors_exit_2(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert message in err


@pytest.mark.parametrize(
    "argv", [["run", SUPPORT], ["run", SUPPORT, "--agent", REFERENCE, "--timeout", "0"], []]
)
def test_cli_usage_errors_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


def test_cli_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", SUPPORT, POLICY]) == 0
    out = capsys.readouterr().out
    assert f"{SUPPORT}: OK (6 case(s))" in out
    assert f"{POLICY}: OK (3 case(s))" in out

    bad = tmp_path / "bad.yaml"
    bad.write_text("cases:\n  - id: a\n    inptu: typo\n", encoding="utf-8")
    assert main(["validate", str(bad)]) == 2
    assert "'input' is required" in capsys.readouterr().err


def test_module_entry_point_reports_version() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "david_agent_evals", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == f"david-evals {david_agent_evals.__version__}"


# --- reports ----------------------------------------------------------------


def test_text_report_without_usage_says_cost_is_unavailable() -> None:
    suite = make_suite({"id": "a", "input": "x"})
    text = render_text(
        [EvalRunner(ScriptedAgent("s", {"x": AgentResponse("ok")})).run_suite(suite)]
    )
    assert "Cost        not available (no case reported usable usage data)" in text
    assert "PASS" in text


def test_json_report_tolerates_non_json_tool_arguments() -> None:
    when = datetime(2026, 1, 2, 3, 4, 5)
    agent = ScriptedAgent(
        "s", {"x": AgentResponse("ok", tool_calls=[ToolCall("schedule", {"at": when})])}
    )
    result = EvalRunner(agent).run_suite(make_suite({"id": "a", "input": "x"}))
    report = json.loads(to_json([result]))
    assert report["suites"][0]["cases"][0]["tool_calls"][0]["arguments"]["at"] == str(when)
    assert result.cases[0].status is CaseStatus.PASSED
