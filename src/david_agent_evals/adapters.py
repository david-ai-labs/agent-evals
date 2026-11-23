"""The adapter interface that connects any agent to the evaluation runner."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from david_agent_evals.types import AgentRequest, AgentResponse


@runtime_checkable
class AgentAdapter(Protocol):
    """Structural interface for an agent under evaluation.

    Implement this for an agent built on any provider or framework: translate
    the :class:`AgentRequest` into a call to your agent, then translate what it
    did into an :class:`AgentResponse`. No base class is required; any object
    with a ``name`` and a ``run`` method qualifies.

    ``run`` may be a regular method or an ``async`` method. Prefer ``async``
    for agents that do network I/O, because the runner can cancel an async call
    that exceeds its timeout, whereas a blocking call can only be abandoned.
    """

    @property
    def name(self) -> str:
        """A short identifier shown in reports."""
        ...

    def run(self, request: AgentRequest) -> AgentResponse | Awaitable[AgentResponse]:
        """Handle one request and describe what the agent did."""
        ...


@dataclass(frozen=True, slots=True)
class FunctionAdapter:
    """Wrap a plain sync or async function as an :class:`AgentAdapter`."""

    name: str
    function: Callable[[AgentRequest], AgentResponse | Awaitable[AgentResponse]]

    def run(self, request: AgentRequest) -> AgentResponse | Awaitable[AgentResponse]:
        return self.function(request)
