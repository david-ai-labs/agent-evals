"""Provider-independent data types exchanged between the runner and an agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single tool invocation made by an agent while handling a request."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True, slots=True)
class Usage:
    """Resource usage reported by the agent's provider, where available.

    Every field is optional. Adapters should leave a field as ``None`` rather
    than guess: the cost metric reports "unavailable" instead of inventing a
    number.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True, slots=True)
class AgentRequest:
    """What the runner hands to an agent for one evaluation case.

    ``input`` is whatever the case file defines: usually a string, but any
    JSON-compatible value (for example a list of chat messages) is passed
    through unchanged. ``context`` carries optional case-specific data.
    """

    input: Any
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """What an agent returns: its final output plus an execution trace."""

    output: str
    tool_calls: Sequence[ToolCall] = ()
    usage: Usage | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
