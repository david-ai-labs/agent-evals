from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from david_agent_evals import CaseValidationError, load_suite, parse_suite
from david_agent_evals.cases import ArgumentMatchMode, ToolMatchMode

SUITE: dict[str, Any] = {
    "name": "demo",
    "description": "A demo suite",
    "defaults": {"timeout_seconds": 5},
    "pricing": {"input_per_million_usd": 1, "output_per_million_usd": 2},
    "cases": [
        {
            "id": "first",
            "input": "hello",
            "description": "says hello",
            "context": {"locale": "en"},
            "tags": ["smoke"],
            "timeout_seconds": 1.5,
            "expect": {
                "output": {"contains": "hi", "not_contains": ["bye"], "ignore_case": True},
                "tool_calls": {
                    "match": "in_order",
                    "arguments_match": "exact",
                    "calls": [{"name": "greet", "arguments": {"who": "world"}}],
                    "forbidden": "shutdown",
                },
                "max_latency_seconds": 2,
                "max_cost_usd": 0.01,
            },
        },
        {"id": "second", "input": [{"role": "user", "content": "hi"}]},
    ],
}

YAML_SUITE = """
name: demo
defaults:
  timeout_seconds: 5
cases:
  - id: first
    input: hello
    expect:
      output:
        contains: hi
      tool_calls:
        calls:
          - name: greet
            arguments: {who: world}
"""


def test_parse_suite_reads_every_field() -> None:
    suite = parse_suite(SUITE, source="demo.json")

    assert (suite.name, suite.description, suite.source) == ("demo", "A demo suite", "demo.json")
    assert suite.default_timeout_seconds == 5
    assert suite.pricing is not None
    assert suite.pricing.cost_usd(1_000_000, 500_000) == pytest.approx(2.0)

    first, second = suite.cases
    assert first.context == {"locale": "en"}
    assert first.tags == ("smoke",)
    assert first.timeout_seconds == 1.5
    assert first.expect.output is not None
    assert first.expect.output.contains == ("hi",)
    assert first.expect.output.not_contains == ("bye",)
    assert first.expect.output.ignore_case is True
    tools = first.expect.tool_calls
    assert tools is not None
    assert tools.mode is ToolMatchMode.IN_ORDER
    assert tools.arguments_mode is ArgumentMatchMode.EXACT
    assert tools.calls is not None
    assert tools.calls[0].arguments == {"who": "world"}
    assert tools.forbidden == ("shutdown",)
    assert first.expect.max_latency_seconds == 2
    assert first.expect.max_cost_usd == 0.01

    assert second.input == [{"role": "user", "content": "hi"}]
    assert second.expect.output is None
    assert second.expect.tool_calls is None


def test_tool_call_defaults_are_exact_order_and_subset_arguments() -> None:
    suite = parse_suite(
        {"cases": [{"id": "a", "input": "x", "expect": {"tool_calls": {"calls": []}}}]}
    )
    tools = suite.cases[0].expect.tool_calls
    assert tools is not None
    assert tools.calls == ()
    assert tools.mode is ToolMatchMode.EXACT
    assert tools.arguments_mode is ArgumentMatchMode.SUBSET


def test_load_suite_json_and_yaml_agree(tmp_path: Path) -> None:
    yaml_file = tmp_path / "suite.yaml"
    yaml_file.write_text(YAML_SUITE, encoding="utf-8")
    json_file = tmp_path / "suite.json"
    json_file.write_text(
        json.dumps(
            {
                "name": "demo",
                "defaults": {"timeout_seconds": 5},
                "cases": [
                    {
                        "id": "first",
                        "input": "hello",
                        "expect": {
                            "output": {"contains": "hi"},
                            "tool_calls": {
                                "calls": [{"name": "greet", "arguments": {"who": "world"}}]
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    from_yaml = load_suite(yaml_file)
    from_json = load_suite(json_file)

    assert from_yaml.cases == from_json.cases
    assert from_yaml.default_timeout_seconds == from_json.default_timeout_seconds == 5
    assert from_yaml.source == str(yaml_file)


def test_suite_name_defaults_to_file_stem(tmp_path: Path) -> None:
    file = tmp_path / "checkout_flow.yml"
    file.write_text("cases:\n  - id: a\n    input: x\n", encoding="utf-8")
    assert load_suite(file).name == "checkout_flow"


def test_select_filters_by_tag() -> None:
    suite = parse_suite(
        {
            "cases": [
                {"id": "a", "input": "x", "tags": ["fast"]},
                {"id": "b", "input": "x", "tags": ["slow", "refund"]},
                {"id": "c", "input": "x"},
            ]
        }
    )
    assert [case.id for case in suite.select([])] == ["a", "b", "c"]
    assert [case.id for case in suite.select(["refund", "fast"])] == ["a", "b"]
    assert suite.select(["missing"]) == ()


def _case(**overrides: Any) -> dict[str, Any]:
    return {"cases": [{"id": "a", "input": "x", **overrides}]}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], "expected a mapping"),
        ({"cases": []}, "'cases' must be a non-empty list"),
        ({"name": "x"}, "'cases' must be a non-empty list"),
        ({"cases": [{"input": "x"}]}, "'id' is required"),
        ({"cases": [{"id": 7, "input": "x"}]}, "'id' must be a string"),
        ({"cases": [{"id": "a"}]}, "'input' is required"),
        (
            {"cases": [{"id": "a", "input": "x"}, {"id": "a", "input": "y"}]},
            "duplicate case id 'a'",
        ),
        (_case(inptu="typo"), "unknown key(s) 'inptu'"),
        (_case(expect={"outputs": {}}), "unknown key(s) 'outputs'"),
        (_case(timeout_seconds=0), "must be greater than 0"),
        (_case(timeout_seconds="fast"), "'timeout_seconds' must be a number"),
        (_case(timeout_seconds=True), "'timeout_seconds' must be a number"),
        (_case(tags=[1, 2]), "'tags' must be a string or a list of strings"),
        (_case(expect={"output": {}}), "needs at least one of"),
        (_case(expect={"output": {"matches": "("}}), "not a valid regular expression"),
        (_case(expect={"output": {"matches": "a" * 1001}}), "limit 1000"),
        (_case(expect={"output": {"contains": "x", "ignore_case": "yes"}}), "true or false"),
        (_case(expect={"tool_calls": {}}), "needs 'calls', 'forbidden', or both"),
        (_case(expect={"tool_calls": {"calls": [{"arguments": {}}]}}), "'name' is required"),
        (_case(expect={"tool_calls": {"calls": [], "match": "fuzzy"}}), "must be one of: exact"),
        (_case(expect={"max_cost_usd": -1}), "must be at least 0"),
        (
            {"cases": [{"id": "a", "input": "x"}], "pricing": {"input_per_million_usd": 1}},
            "required",
        ),
    ],
)
def test_invalid_suites_are_rejected(data: object, message: str) -> None:
    with pytest.raises(CaseValidationError) as excinfo:
        parse_suite(data, source="bad.yaml")
    assert message in str(excinfo.value)
    assert "bad.yaml" in str(excinfo.value)


def test_validation_error_names_the_case() -> None:
    with pytest.raises(CaseValidationError, match=r"cases\[1\] \('second'\)"):
        parse_suite(
            {"cases": [{"id": "first", "input": "x"}, {"id": "second", "input": "x", "oops": 1}]}
        )


def test_load_suite_rejects_unparseable_files(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(CaseValidationError, match="could not parse"):
        load_suite(broken)

    broken_yaml = tmp_path / "broken.yaml"
    broken_yaml.write_text("cases: [unclosed", encoding="utf-8")
    with pytest.raises(CaseValidationError, match="could not parse"):
        load_suite(broken_yaml)


def test_load_suite_rejects_unknown_extension(tmp_path: Path) -> None:
    file = tmp_path / "suite.toml"
    file.write_text("", encoding="utf-8")
    with pytest.raises(CaseValidationError, match="unsupported file type"):
        load_suite(file)


def test_load_suite_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_suite(tmp_path / "absent.yaml")
