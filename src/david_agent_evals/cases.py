"""Evaluation case and suite definitions, loaded from JSON or YAML files.

Loading is strict: unknown keys, wrong types and duplicate case ids are
rejected with an error that names the file and the case, so a typo in an
expectation cannot silently turn a check into a no-op.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import yaml

from david_agent_evals.regex import pattern_problem


class CaseValidationError(ValueError):
    """Raised when a case file is malformed."""


class ToolMatchMode(StrEnum):
    """How the expected tool calls are compared with the calls an agent made."""

    EXACT = "exact"
    """Same calls, same order, nothing extra."""
    IN_ORDER = "in_order"
    """Expected calls appear in order; other calls may be interleaved."""
    ANY_ORDER = "any_order"
    """Each expected call appears somewhere; other calls are allowed."""


class ArgumentMatchMode(StrEnum):
    """How expected tool arguments are compared with actual arguments."""

    SUBSET = "subset"
    """Every expected argument must be present and equal; extras are allowed."""
    EXACT = "exact"
    """Arguments must be equal, with no extras."""


@dataclass(frozen=True, slots=True)
class ExpectedToolCall:
    name: str
    arguments: Mapping[str, Any] | None = None
    """``None`` means the arguments are not checked."""


@dataclass(frozen=True, slots=True)
class ToolCallExpectation:
    calls: tuple[ExpectedToolCall, ...] | None = None
    """``None`` means only ``forbidden`` is checked. An empty tuple with
    ``mode=EXACT`` asserts that the agent calls no tools at all."""
    mode: ToolMatchMode = ToolMatchMode.EXACT
    arguments_mode: ArgumentMatchMode = ArgumentMatchMode.SUBSET
    forbidden: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OutputExpectation:
    equals: str | None = None
    contains: tuple[str, ...] = ()
    not_contains: tuple[str, ...] = ()
    matches: str | None = None
    ignore_case: bool = False


@dataclass(frozen=True, slots=True)
class Expectation:
    output: OutputExpectation | None = None
    tool_calls: ToolCallExpectation | None = None
    max_latency_seconds: float | None = None
    max_cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class Pricing:
    """User-supplied token prices, used only when an agent reports token
    counts but no cost. The framework ships no provider price tables."""

    input_per_million_usd: float
    output_per_million_usd: float

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_million_usd + output_tokens * self.output_per_million_usd
        ) / 1_000_000


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    input: Any
    description: str = ""
    context: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    timeout_seconds: float | None = None
    expect: Expectation = field(default_factory=Expectation)


@dataclass(frozen=True, slots=True)
class EvalSuite:
    name: str
    cases: tuple[EvalCase, ...]
    description: str = ""
    default_timeout_seconds: float | None = None
    pricing: Pricing | None = None
    source: str | None = None

    def select(self, tags: Collection[str]) -> tuple[EvalCase, ...]:
        """Cases carrying at least one of ``tags``; all cases if ``tags`` is empty."""
        if not tags:
            return self.cases
        wanted = set(tags)
        return tuple(case for case in self.cases if wanted.intersection(case.tags))


def load_suite(path: str | Path) -> EvalSuite:
    """Load a suite from a ``.json``, ``.yaml`` or ``.yml`` file."""
    file = Path(path)
    suffix = file.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise CaseValidationError(
            f"{file}: unsupported file type '{suffix}' (expected .json, .yaml or .yml)"
        )
    text = file.read_text(encoding="utf-8")
    try:
        data = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CaseValidationError(f"{file}: could not parse file: {exc}") from exc
    return parse_suite(data, source=str(file), default_name=file.stem)


def parse_suite(
    data: object, *, source: str = "<memory>", default_name: str = "unnamed-suite"
) -> EvalSuite:
    """Validate already-parsed JSON/YAML data and build an :class:`EvalSuite`."""
    root = _Node(data, source)
    name = root.opt_str("name") or default_name
    description = root.opt_str("description") or ""

    default_timeout: float | None = None
    defaults = root.child("defaults")
    if defaults is not None:
        default_timeout = defaults.opt_number("timeout_seconds", minimum=0, exclusive=True)
        defaults.finish()

    pricing: Pricing | None = None
    pricing_node = root.child("pricing")
    if pricing_node is not None:
        pricing = Pricing(
            input_per_million_usd=pricing_node.req_number("input_per_million_usd", minimum=0),
            output_per_million_usd=pricing_node.req_number("output_per_million_usd", minimum=0),
        )
        pricing_node.finish()

    raw_cases = root.opt_list("cases")
    if not raw_cases:
        raise CaseValidationError(f"{source}: 'cases' must be a non-empty list")
    root.finish()

    cases = tuple(_parse_case(raw, f"{source}: cases[{i}]") for i, raw in enumerate(raw_cases))
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise CaseValidationError(f"{source}: duplicate case id '{case.id}'")
        seen.add(case.id)

    return EvalSuite(
        name=name,
        cases=cases,
        description=description,
        default_timeout_seconds=default_timeout,
        pricing=pricing,
        source=source,
    )


def _parse_case(raw: object, where: str) -> EvalCase:
    node = _Node(raw, where)
    case_id = node.req_str("id")
    node.where = f"{where} ('{case_id}')"
    case_input = node.take("input")
    if case_input is None:
        raise CaseValidationError(f"{node.where}: 'input' is required")
    case = EvalCase(
        id=case_id,
        input=case_input,
        description=node.opt_str("description") or "",
        context=node.opt_mapping("context") or {},
        tags=node.str_list("tags"),
        timeout_seconds=node.opt_number("timeout_seconds", minimum=0, exclusive=True),
        expect=_parse_expectation(node.child("expect")),
    )
    node.finish()
    return case


def _parse_expectation(node: _Node | None) -> Expectation:
    if node is None:
        return Expectation()
    expectation = Expectation(
        output=_parse_output(node.child("output")),
        tool_calls=_parse_tool_calls(node.child("tool_calls")),
        max_latency_seconds=node.opt_number("max_latency_seconds", minimum=0, exclusive=True),
        max_cost_usd=node.opt_number("max_cost_usd", minimum=0),
    )
    node.finish()
    return expectation


def _parse_output(node: _Node | None) -> OutputExpectation | None:
    if node is None:
        return None
    pattern = node.opt_str("matches")
    if pattern is not None and (problem := pattern_problem(pattern)) is not None:
        raise CaseValidationError(f"{node.where}: 'matches': {problem}")
    output = OutputExpectation(
        equals=node.opt_str("equals"),
        contains=node.str_list("contains"),
        not_contains=node.str_list("not_contains"),
        matches=pattern,
        ignore_case=node.opt_bool("ignore_case", default=False),
    )
    node.finish()
    if output.equals is None and not output.contains and not output.not_contains and not pattern:
        raise CaseValidationError(
            f"{node.where}: needs at least one of 'equals', 'contains', 'not_contains', 'matches'"
        )
    return output


def _parse_tool_calls(node: _Node | None) -> ToolCallExpectation | None:
    if node is None:
        return None
    raw_calls = node.opt_list("calls")
    calls: tuple[ExpectedToolCall, ...] | None = None
    if raw_calls is not None:
        parsed = []
        for i, raw in enumerate(raw_calls):
            call_node = _Node(raw, f"{node.where}.calls[{i}]")
            parsed.append(
                ExpectedToolCall(
                    name=call_node.req_str("name"), arguments=call_node.opt_mapping("arguments")
                )
            )
            call_node.finish()
        calls = tuple(parsed)
    expectation = ToolCallExpectation(
        calls=calls,
        mode=_enum(node, "match", ToolMatchMode, ToolMatchMode.EXACT),
        arguments_mode=_enum(node, "arguments_match", ArgumentMatchMode, ArgumentMatchMode.SUBSET),
        forbidden=node.str_list("forbidden"),
    )
    node.finish()
    if expectation.calls is None and not expectation.forbidden:
        raise CaseValidationError(f"{node.where}: needs 'calls', 'forbidden', or both")
    return expectation


_E = TypeVar("_E", bound=StrEnum)


def _enum(node: _Node, key: str, enum: type[_E], default: _E) -> _E:
    value = node.opt_str(key)
    if value is None:
        return default
    try:
        return enum(value)
    except ValueError:
        allowed = ", ".join(member.value for member in enum)
        raise CaseValidationError(
            f"{node.where}: '{key}' must be one of: {allowed} (got '{value}')"
        ) from None


def _kind(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "a mapping"
    if isinstance(value, list):
        return "a list"
    return f"{type(value).__name__} {value!r}"


class _Node:
    """A mapping under validation: knows where it is and which keys were read."""

    def __init__(self, data: object, where: str) -> None:
        if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
            raise CaseValidationError(
                f"{where}: expected a mapping with string keys, got {_kind(data)}"
            )
        self._data: dict[str, Any] = data
        self._read: set[str] = set()
        self.where = where

    def take(self, key: str) -> Any:
        self._read.add(key)
        return self._data.get(key)

    def _wrong(self, key: str, expected: str, value: object) -> CaseValidationError:
        return CaseValidationError(f"{self.where}: '{key}' must be {expected}, got {_kind(value)}")

    def opt_str(self, key: str) -> str | None:
        value = self.take(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise self._wrong(key, "a string (quote it if it looks like a number)", value)
        return value

    def req_str(self, key: str) -> str:
        value = self.opt_str(key)
        if value is None or not value.strip():
            raise CaseValidationError(f"{self.where}: '{key}' is required and cannot be empty")
        return value

    def opt_number(self, key: str, *, minimum: float, exclusive: bool = False) -> float | None:
        value = self.take(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self._wrong(key, "a number", value)
        if value < minimum or (exclusive and value == minimum):
            bound = "greater than" if exclusive else "at least"
            raise CaseValidationError(f"{self.where}: '{key}' must be {bound} {minimum:g}")
        return float(value)

    def req_number(self, key: str, *, minimum: float) -> float:
        value = self.opt_number(key, minimum=minimum)
        if value is None:
            raise CaseValidationError(f"{self.where}: '{key}' is required")
        return value

    def opt_bool(self, key: str, *, default: bool) -> bool:
        value = self.take(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise self._wrong(key, "true or false", value)
        return value

    def str_list(self, key: str) -> tuple[str, ...]:
        """A list of strings; a single string is accepted as a one-item list."""
        value = self.take(key)
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise self._wrong(key, "a string or a list of strings", value)
        return tuple(value)

    def opt_list(self, key: str) -> list[Any] | None:
        value = self.take(key)
        if value is None:
            return None
        if not isinstance(value, list):
            raise self._wrong(key, "a list", value)
        return value

    def opt_mapping(self, key: str) -> dict[str, Any] | None:
        value = self.take(key)
        if value is None:
            return None
        if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
            raise self._wrong(key, "a mapping with string keys", value)
        return value

    def child(self, key: str) -> _Node | None:
        value = self.take(key)
        return None if value is None else _Node(value, f"{self.where}.{key}")

    def finish(self) -> None:
        """Reject keys nothing asked for, which are almost always typos."""
        unknown = sorted(set(self._data) - self._read)
        if unknown:
            allowed = ", ".join(sorted(self._read))
            raise CaseValidationError(
                f"{self.where}: unknown key(s) {', '.join(map(repr, unknown))} (allowed: {allowed})"
            )
