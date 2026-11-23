# Architecture

## Project

DAVID Agent Evaluations

## Purpose

Develop methods for evaluating agent behavior, tool execution, reliability, latency, cost, and failure recovery.

## Architecture Status

This document describes v0.1, the first implementation. v0.1 covers task
success, tool-call correctness, latency and cost for single-turn cases. Failure
recovery is not yet evaluated. Sections listed under
[Planned Documentation](#planned-documentation) are still to be written.

## System Context

The framework is a Python library and command-line tool that runs on the
machine of whoever is evaluating an agent, typically a developer laptop or a
CI job. It has three external touchpoints:

- **Case files** (JSON or YAML) written by the evaluator.
- **An agent adapter**, Python code supplied by the evaluator, which calls the
  agent under test. The agent itself, and any model provider or tools behind
  it, are outside the framework.
- **Reports**: text on stdout and an optional JSON results file.

The framework makes no network calls and stores nothing beyond the results
file it is asked to write.

## Components and Responsibilities

| Module | Responsibility |
| --- | --- |
| `types.py` | `AgentRequest`, `AgentResponse`, `ToolCall`, `Usage`: immutable, provider-independent data exchanged with an agent. |
| `adapters.py` | `AgentAdapter`, a structural protocol (`name`, `run`), and `FunctionAdapter`. |
| `cases.py` | `EvalSuite`, `EvalCase` and expectation models; strict loading and validation of JSON/YAML. |
| `runner.py` | `EvalRunner`: invokes the adapter per case under a timeout and applies metrics. |
| `metrics.py` | The `Metric` protocol; `TaskSuccess`, `ToolCallCorrectness`, `Latency`, `Cost`. |
| `regex.py` | Runs a regular-expression search in a child process with a hard time limit. |
| `results.py` | `CaseResult`, `SuiteResult`, and summary aggregation. |
| `report.py` | Text and JSON rendering. |
| `loader.py` | Resolves an `--agent` specification (file path or module path) to an adapter. |
| `cli.py` | Argument parsing, exit codes. |
| `mock.py` | Deterministic agents for tests and examples. |

Dependencies point one way: `cli` → `loader`, `runner`, `report` → `metrics`,
`results` → `cases`, `types`. Nothing below the CLI reads arguments, prints,
or exits, so every layer is usable as a library.

## Data Flow

1. `cases.load_suite` parses a file and validates it into an immutable
   `EvalSuite`. Invalid files stop here.
2. `loader.load_agent` imports the evaluator's adapter.
3. For each selected case, `EvalRunner.run_case`:
   1. resolves the timeout (case, then suite default, then runner default);
   2. calls `adapter.run(AgentRequest)` on a daemon worker thread and waits up
      to the timeout;
   3. on a response, builds a `MetricContext` and asks each metric for a
      `MetricResult`;
   4. returns a `CaseResult` with status `passed`, `failed`, `error` or
      `timeout`.
4. `SuiteResult.summary` aggregates the case results.
5. `report` renders text and JSON.

## Failure Modes

| Failure | Handling |
| --- | --- |
| Malformed case file | `CaseValidationError` naming the file, case and key. CLI exits 2. Nothing runs. |
| Agent cannot be imported or is not an adapter | `AgentLoadError`. CLI exits 2. |
| Adapter raises | Case status `error` with the exception type and message. The run continues. |
| Adapter returns the wrong type | Case status `error`. The run continues. |
| Adapter exceeds its timeout | Case status `timeout`. An async call is cancelled; a blocking call is abandoned on its daemon thread. The run continues. |
| Adapter raises its own `TimeoutError` | Case status `error`, not `timeout`: the runner's deadline did not expire. |
| A metric raises | That metric is recorded as failed with the exception; the case fails. The run continues. |
| No usage data, no cost limit | Cost is null and informational. |
| No usage data, `max_cost_usd` set | The `cost` metric fails with the reason; the case fails. |
| A `matches` search overruns its time limit | The search process is killed; `task_success` fails with the reason. The run continues. |
| Tag filter selects no cases | CLI exits 2, so a mistyped tag cannot produce an empty, passing run. |

## Trust Boundaries

- **Adapter code is trusted and runs in-process** with the evaluator's
  privileges. `--agent` imports and executes the named file or module. The
  framework does not sandbox it.
- **Case files are treated as data.** YAML is parsed with `yaml.safe_load`.
  `expect.output.matches` is a Python regular expression, and Python's engine
  can take exponential time on a hostile or careless pattern. Patterns are
  limited to 1000 characters, and every search runs in a child interpreter
  (`python -I -S`) that is killed after a time limit, 2 seconds by default, so
  a pattern cannot block a run indefinitely. The child receives only the
  pattern, the output text and a flag, as JSON on stdin.
- **Agent output is untrusted data.** It is compared, stored and printed, never
  executed. Outputs and tool arguments are written to the results file in
  full; if an agent handles sensitive data, the results file contains it.

## Security Considerations

- No credentials are read, required or stored by the framework. Adapters that
  need API keys obtain them themselves.
- The repository's examples use an in-memory agent with invented order data.
- Error results record the exception type and message, not a traceback.

## Design Decisions

**The adapter is a structural protocol, not a base class.** An agent built on
any SDK can be adapted without inheriting from or importing framework classes
beyond the request and response types. This keeps the framework free of
provider dependencies.

**Adapters may be sync or async.** Agent SDKs come in both styles. Requiring
one would force wrappers on half of them.

**Each call runs on a daemon thread.** This gives one timeout mechanism for
sync and async adapters, lets the runner be called from inside a running event
loop, and means a hung agent cannot stall the run or block process exit. The
cost is that a blocking call cannot be interrupted, only abandoned. Running
adapters in subprocesses would allow a hard kill but would require adapters
and responses to be picklable and was judged out of scope for v0.1.

**Cases run sequentially.** It keeps latency measurements free of contention
between cases and keeps results in a predictable order. Concurrency is a
candidate for a later release.

**Case validation is strict.** Unknown keys are errors. In an evaluation tool a
silently ignored expectation is worse than a refused file, because it produces
a passing result that checked nothing.

**A requirement that cannot be checked fails.** A metric reports
`passed: null` only when the case set no requirement for it, in which case its
value is informational. When the case did set a requirement and the framework
cannot evaluate it, the metric fails with the reason: `max_cost_usd` with no
cost data, or a `matches` search that overran its time limit. The alternative,
reporting "not evaluated" and letting the case pass, would mean a green result
for a requirement nobody verified. A separate "inconclusive" case status was
considered and left out of v0.1: it would add a fifth status for every
consumer of the results to handle, and the failure reason already says what
to fix.

**Regular-expression searches run in a child process.** Python's `re` has no
timeout and holds the interpreter lock for the whole search, so a thread
cannot bound it; only killing a process can. The alternatives were rejected:
restricting patterns to a provably safe subset needs a pattern analyser that
is easy to get wrong and would reject many harmless patterns, and a
linear-time engine would add a compiled dependency. The cost is one process
start per `matches` check.

**Cost is never estimated.** The framework ships no price tables, which would
go stale, and reports cost as unknown unless the adapter or the suite file
supplies the data.

**Latency statistics cover responding cases only.** The duration of a timed-out
or errored case is not a response time, so it is excluded from the mean and
percentiles. Percentiles use the nearest-rank method.

**PyYAML is the only runtime dependency.** Validation is hand-written against
the small case schema instead of pulling in a schema library.

## Planned Documentation

- Integration points for provider adapters
- Operational requirements
- Architecture decision records
