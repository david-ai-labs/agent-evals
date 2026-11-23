"""Deterministic example agents for the sample evaluation suites.

``SupportAgent`` is a tiny rule-based customer-support agent with three
in-memory "tools". It calls no model and no network, so the examples run
anywhere without API keys and give the same result every time.

Two instances are exported:

* ``reference_agent`` follows its refund policy and passes every sample case.
* ``faulty_agent`` has four deliberate defects, so you can see how failures,
  errors and timeouts are reported.

The token counts it reports are simulated (whitespace-separated word counts).
They exist only to exercise the cost metric and say nothing about any real
model.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from david_agent_evals import AgentRequest, AgentResponse, ToolCall, Usage

AUTO_REFUND_LIMIT_USD = 500.0

ORDERS: dict[str, dict[str, Any]] = {
    "A-1001": {"status": "delivered", "total_usd": 42.5, "refundable": True},
    "A-1002": {"status": "in transit", "total_usd": 18.0, "refundable": False},
    "A-1003": {"status": "delivered", "total_usd": 900.0, "refundable": True},
}

_ORDER_ID = re.compile(r"\b[A-Z]-\d{4}\b")


@dataclass(frozen=True, slots=True)
class SupportAgent:
    name: str
    skip_lookup: bool = False
    """Defect: refund without first looking the order up."""
    ignore_refund_policy: bool = False
    """Defect: refund ineligible and over-limit orders."""
    slow_tracking: bool = False
    """Defect: order-status answers for known orders take two seconds."""
    escalation_outage: bool = False
    """Defect: escalating to a human raises an exception."""

    def run(self, request: AgentRequest) -> AgentResponse:
        text = str(request.input)
        lowered = text.lower()
        found = _ORDER_ID.search(text)
        order_id = found.group() if found else None
        calls: list[ToolCall] = []

        if any(phrase in lowered for phrase in ("human", "representative", "real person")):
            output = self._escalate(calls, "customer_request")
        elif "refund" in lowered or "money back" in lowered:
            output = self._refund(calls, order_id)
        elif any(phrase in lowered for phrase in ("where is", "status", "track")):
            output = self._status(calls, order_id)
        else:
            output = "I can help with order status and refunds. What do you need?"

        usage = Usage(input_tokens=len(text.split()), output_tokens=len(output.split()))
        return AgentResponse(output=output, tool_calls=calls, usage=usage)

    def _lookup(self, calls: list[ToolCall], order_id: str) -> dict[str, Any] | None:
        calls.append(ToolCall("lookup_order", {"order_id": order_id}))
        return ORDERS.get(order_id)

    def _escalate(self, calls: list[ToolCall], reason: str) -> str:
        if self.escalation_outage:
            raise RuntimeError("ticketing backend unavailable")
        calls.append(ToolCall("escalate_to_human", {"reason": reason}))
        return "I'm connecting you with a human agent now."

    def _status(self, calls: list[ToolCall], order_id: str | None) -> str:
        if order_id is None:
            return "Please share your order number so I can check its status."
        order = self._lookup(calls, order_id)
        if order is None:
            return f"I could not find order {order_id}. Please check the number."
        if self.slow_tracking:
            time.sleep(2.0)
        return f"Order {order_id} is {order['status']}."

    def _refund(self, calls: list[ToolCall], order_id: str | None) -> str:
        if order_id is None:
            return "I can help with that. What is your order number?"
        order = ORDERS.get(order_id) if self.skip_lookup else self._lookup(calls, order_id)
        if order is None:
            return f"I could not find order {order_id}. Please check the number."
        amount = float(order["total_usd"])
        if not self.ignore_refund_policy:
            if not order["refundable"]:
                status = order["status"]
                return f"Order {order_id} is not eligible for a refund because it is {status}."
            if amount > AUTO_REFUND_LIMIT_USD:
                self._escalate(calls, "refund_over_limit")
                return (
                    f"A refund of ${amount:.2f} for order {order_id} needs approval, "
                    "so I've passed it to a human agent."
                )
        calls.append(ToolCall("issue_refund", {"order_id": order_id, "amount_usd": amount}))
        return f"I've issued a refund of ${amount:.2f} for order {order_id}."


reference_agent = SupportAgent(name="reference-support-agent")

faulty_agent = SupportAgent(
    name="faulty-support-agent",
    skip_lookup=True,
    ignore_refund_policy=True,
    slow_tracking=True,
    escalation_outage=True,
)
