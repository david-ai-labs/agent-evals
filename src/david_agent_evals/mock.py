"""Deterministic mock agents for tests, examples and trying out the framework.

Nothing here calls a model or the network.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

from david_agent_evals.types import AgentRequest, AgentResponse


class UnscriptedInputError(LookupError):
    """Raised by :class:`ScriptedAgent` for an input it has no turn for."""


@dataclass(frozen=True, slots=True)
class ScriptedTurn:
    """What a :class:`ScriptedAgent` does for one input.

    It waits ``delay_seconds``, then raises ``raises`` if set, otherwise
    returns ``response``.
    """

    response: AgentResponse | None = None
    raises: Exception | None = None
    delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        if (self.response is None) == (self.raises is None):
            raise ValueError("a ScriptedTurn needs exactly one of 'response' or 'raises'")


@dataclass(frozen=True, slots=True)
class ScriptedAgent:
    """Replays a fixed turn for each known input string.

    Useful for testing metrics and reports against an exactly known agent
    behaviour, including failures, errors and slow responses.
    """

    name: str
    script: Mapping[str, ScriptedTurn | AgentResponse]
    default: ScriptedTurn | AgentResponse | None = None

    def run(self, request: AgentRequest) -> AgentResponse:
        turn = self.script.get(request.input) if isinstance(request.input, str) else None
        turn = turn or self.default
        if turn is None:
            raise UnscriptedInputError(f"no scripted turn for input {request.input!r}")
        if isinstance(turn, AgentResponse):
            return turn
        if turn.delay_seconds:
            time.sleep(turn.delay_seconds)
        if turn.raises is not None:
            raise turn.raises
        assert turn.response is not None  # guaranteed by ScriptedTurn.__post_init__
        return turn.response


@dataclass(frozen=True, slots=True)
class EchoAgent:
    """Returns its input as its output and calls no tools."""

    name: str = "echo-agent"

    def run(self, request: AgentRequest) -> AgentResponse:
        return AgentResponse(output=str(request.input))
