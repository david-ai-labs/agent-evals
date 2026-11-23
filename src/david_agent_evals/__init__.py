"""DAVID Agent Evaluations: a small framework for evaluating AI agent reliability."""

from david_agent_evals.adapters import AgentAdapter, FunctionAdapter
from david_agent_evals.cases import (
    CaseValidationError,
    EvalCase,
    EvalSuite,
    Expectation,
    ExpectedToolCall,
    OutputExpectation,
    Pricing,
    ToolCallExpectation,
    load_suite,
    parse_suite,
)
from david_agent_evals.metrics import Metric, MetricContext, MetricResult, default_metrics
from david_agent_evals.results import CaseResult, CaseStatus, SuiteResult, SuiteSummary
from david_agent_evals.runner import EvalRunner
from david_agent_evals.types import AgentRequest, AgentResponse, ToolCall, Usage

__version__ = "0.1.0"

__all__ = [
    "AgentAdapter",
    "AgentRequest",
    "AgentResponse",
    "CaseResult",
    "CaseStatus",
    "CaseValidationError",
    "EvalCase",
    "EvalRunner",
    "EvalSuite",
    "Expectation",
    "ExpectedToolCall",
    "FunctionAdapter",
    "Metric",
    "MetricContext",
    "MetricResult",
    "OutputExpectation",
    "Pricing",
    "SuiteResult",
    "SuiteSummary",
    "ToolCall",
    "ToolCallExpectation",
    "Usage",
    "__version__",
    "default_metrics",
    "load_suite",
    "parse_suite",
]
