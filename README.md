# DAVID Agent Evaluations

[![CI](https://github.com/david-ai-labs/agent-evals/actions/workflows/ci.yml/badge.svg)](https://github.com/david-ai-labs/agent-evals/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Evaluation frameworks for reliable enterprise AI agents.

This repository contains a small, provider-independent Python framework for
evaluating AI agent reliability.

You describe what an agent should do in a JSON or YAML file: the input, the
output you expect, the tool calls it should and should not make, and optional
latency and cost limits. The framework runs each case against your agent with
a timeout, scores the response, and produces a readable report and a JSON
results file. It exits non-zero when anything fails, so it can gate a CI job.

**Status: v0.1, alpha.** This is a first release with a deliberately small
scope. Read [Limitations](#limitations) before relying on it.

## Installation

Requires Python 3.11 or newer. The only runtime dependency is PyYAML.

```bash
git clone https://github.com/david-ai-labs/agent-evals.git
cd agent-evals
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e .
```

The package is not published to PyPI.

## Quick start

The repository ships a deterministic example agent and two evaluation suites.
They call no model and no network, so no API keys are needed.

```bash
david-evals run examples/cases/support.yaml --agent examples/agents.py:reference_agent
```

```text
DAVID Agent Evaluations v0.1.0
Suite: customer-support  (examples/cases/support.yaml)
Agent: reference-support-agent

  PASS     refund-eligible-order           0.03ms   $0.000025
  PASS     refund-ineligible-order         0.02ms   $0.000030
  PASS     order-status                    0.02ms   $0.000015
  PASS     unknown-order                   0.01ms   $0.000026
  PASS     refund-missing-order-number     0.01ms   $0.000025
  PASS     escalation-request              0.01ms   $0.000025

Summary
  Cases       6 total: 6 passed, 0 failed, 0 errors, 0 timeouts
  Pass rate   100.0%
  Latency     mean 0.01ms, p50 0.01ms, p95 0.03ms, max 0.03ms (6 responding case(s))
  Cost        $0.000146 total (6 of 6 case(s) reported usage)
  Duration    0.80ms

Checks (over the cases where the agent responded and the check applied)
  task_success           6/6 passed
  tool_call_correctness  6/6 passed, mean score 1.00
  latency                1/1 passed
```

Latency figures vary from run to run. The cost column is computed from the
example agent's simulated token counts and placeholder prices in the suite
file; it demonstrates the mechanism and is not a measurement of any real model.

Now run the same suite against `faulty_agent`, the same agent with four
deliberate defects, and write the JSON results:

```bash
david-evals run examples/cases/support.yaml --agent examples/agents.py:faulty_agent --output results.json
```

```text
  FAIL     refund-eligible-order           0.06ms   $0.000025
           tool_call_correctness: matched 0 of 2 expected call(s) (exact); agent called: issue_refund
  FAIL     refund-ineligible-order         0.01ms   $0.000022
           task_success: output does not contain 'not eligible'
           tool_call_correctness: called forbidden tool(s): issue_refund; agent called: issue_refund
  TIMEOUT  order-status                  505.13ms           -
           no response within the 0.5s timeout
  PASS     unknown-order                   0.06ms   $0.000026
  PASS     refund-missing-order-number     0.01ms   $0.000025
  ERROR    escalation-request              0.01ms           -
           RuntimeError: ticketing backend unavailable

Summary
  Cases       6 total: 2 passed, 2 failed, 1 errors, 1 timeouts
  Pass rate   33.3%
```

The command exits with status 1, and `results.json` holds every case's output,
tool calls, usage and per-metric verdicts.

## Writing evaluation cases

A suite is one JSON or YAML file. Both formats use the same structure;
[`examples/cases/support.yaml`](examples/cases/support.yaml) and
[`examples/cases/support_policy.json`](examples/cases/support_policy.json) are
complete examples.

```yaml
name: customer-support            # optional; defaults to the file name
defaults:
  timeout_seconds: 2              # optional; applies to cases that set none
pricing:                          # optional; your own token prices, see "Cost"
  input_per_million_usd: 1.00
  output_per_million_usd: 2.00

cases:
  - id: refund-ineligible-order   # required, unique within the file
    input: "Please refund order A-1002."   # required; a string or any JSON value
    description: An order still in transit must not be refunded.
    tags: [refund, policy]
    context: {locale: en}         # passed to the agent unchanged
    timeout_seconds: 5
    expect:
      output:
        contains: not eligible    # a string or a list; all must appear
      tool_calls:
        calls:
          - name: lookup_order
            arguments: {order_id: A-1002}
        forbidden: [issue_refund]
      max_latency_seconds: 1.0
      max_cost_usd: 0.01
```

Everything under `expect` is optional. A case with no `expect` passes if the
agent responds without an error or timeout.

| Key | Meaning |
| --- | --- |
| `expect.output.equals` | Output equals this text, ignoring leading and trailing whitespace. |
| `expect.output.contains` | Output contains every listed string. |
| `expect.output.not_contains` | Output contains none of the listed strings. |
| `expect.output.matches` | A Python regular expression found anywhere in the output. |
| `expect.output.ignore_case` | Make the output checks case-insensitive. Default `false`. |
| `expect.tool_calls.calls` | The expected tool calls. An entry without `arguments` matches any arguments. |
| `expect.tool_calls.match` | `exact` (default): same calls, same order, nothing extra. `in_order`: expected calls appear in order, other calls may be interleaved. `any_order`: each expected call appears somewhere. |
| `expect.tool_calls.arguments_match` | `subset` (default): the listed arguments must be present and equal; extra arguments are allowed. `exact`: no extra arguments. Nested mappings follow the same mode; lists must be equal. |
| `expect.tool_calls.forbidden` | Tool names the agent must not call. |
| `expect.max_latency_seconds` | Fail the case if the agent takes longer. |
| `expect.max_cost_usd` | Fail the case if the known cost is higher. |

`calls: []` with the default `exact` match asserts that the agent calls no tools.

Case files are validated strictly. Unknown keys, wrong types and duplicate ids
are rejected with a message naming the file and case, so a typo cannot quietly
disable a check. Check files without running an agent:

```bash
david-evals validate examples/cases/support.yaml examples/cases/support_policy.json
```

## Connecting your agent

An adapter is any object with a `name` and a `run` method that takes an
`AgentRequest` and returns an `AgentResponse`. There is no base class and no
dependency on any provider SDK: the adapter is where you translate between
your agent and the framework.

```python
# my_agents.py
from david_agent_evals import AgentRequest, AgentResponse, ToolCall, Usage


class MyAgent:
    name = "my-agent"

    async def run(self, request: AgentRequest) -> AgentResponse:
        result = await call_my_agent(request.input)  # your code
        return AgentResponse(
            output=result.final_text,
            tool_calls=[ToolCall(c.name, c.arguments) for c in result.tool_calls],
            usage=Usage(
                input_tokens=result.input_tokens,  # leave as None if unknown
                output_tokens=result.output_tokens,
            ),
        )
```

```bash
david-evals run cases.yaml --agent my_agents.py:MyAgent
```

`--agent` accepts `path/to/file.py:attribute` or `package.module:attribute`.
The attribute can be an adapter instance, a class, or a zero-argument factory.
`run` can be a regular or an `async` method; prefer `async` for agents that do
network I/O, because an async call is cancelled when it times out, while a
blocking call can only be abandoned.

`--agent` imports and executes the file or module you name. Only point it at
code you trust.

## Metrics

| Metric | Applies when | Value | Pass condition |
| --- | --- | --- | --- |
| `task_success` | `expect.output` is set | 1 or 0 | Every output check holds. |
| `tool_call_correctness` | `expect.tool_calls` is set | Fraction of expected calls matched, 0 to 1 | All expected calls matched under the chosen mode, and no forbidden tool called. A forbidden call sets the value to 0. |
| `latency` | Always | Wall-clock seconds of the adapter call | Within `max_latency_seconds`, if set. Otherwise informational. |
| `cost` | Always | USD, or null when unknown | Within `max_cost_usd`, if set and the cost is known. |

A case is `passed` when the agent responds and no metric fails, `failed` when
a metric fails, `error` when the adapter raises or returns something other
than an `AgentResponse`, and `timeout` when it does not respond in time.

### Cost

Cost is reported only where the agent supplies usage data. In order:

1. If the adapter sets `Usage.cost_usd`, that value is used.
2. Otherwise, if it reports both token counts and the suite file defines
   `pricing`, the cost is computed from those.
3. Otherwise the cost is null. A `max_cost_usd` limit is then marked as not
   evaluated instead of passing or failing, and the report says so.

The framework contains no provider price tables. Prices in a suite file are
yours to supply and keep current.

### Timeouts

The timeout for a case is the first of: the case's `timeout_seconds`, the
suite's `defaults.timeout_seconds`, the `--timeout` flag (default 30 seconds).

## Command-line reference

```text
david-evals run SUITE [SUITE ...] --agent SPEC
                [--timeout SECONDS] [--tag TAG] [--output FILE] [--format {text,json}]
david-evals validate SUITE [SUITE ...]
david-evals --version
```

| Option | Effect |
| --- | --- |
| `--timeout SECONDS` | Fallback timeout for cases with none of their own. |
| `--tag TAG` | Run only cases with this tag. Repeat to select several. |
| `--output FILE` | Write the JSON results to a file as well as printing the report. |
| `--format json` | Print the JSON results to stdout instead of the text report. |

Exit status: `0` every case passed; `1` at least one case failed, errored or
timed out; `2` usage or configuration error (bad case file, agent could not be
loaded, tag filter matched nothing).

## Using it from Python

```python
from david_agent_evals import AgentResponse, EvalRunner, FunctionAdapter, load_suite
from david_agent_evals.report import render_text, write_json

agent = FunctionAdapter("shouting-agent", lambda request: AgentResponse(str(request.input).upper()))
result = EvalRunner(agent, default_timeout_seconds=10).run_suite(load_suite("cases.yaml"))

print(render_text([result]))
write_json([result], "results.json")
```

Custom metrics are objects with a `name` and an `evaluate(context)` method
returning a `MetricResult` (or `None` when not applicable). Pass them to
`EvalRunner(..., metrics=[...])`; include `*default_metrics()` to keep the
built-in ones. Custom metrics are available from Python only, not from the CLI.

`david_agent_evals.mock` provides `ScriptedAgent` and `EchoAgent`,
deterministic agents for testing your own cases and metrics.

## Architecture

```text
case file (.yaml/.json) ──▶ cases.py ──▶ EvalSuite
                                             │
--agent SPEC ──▶ loader.py ──▶ AgentAdapter  │
                                    │        ▼
                                    └──▶ runner.py ──▶ metrics.py
                                             │
                                             ▼
                                        results.py ──▶ report.py ──▶ text / JSON
```

| Module | Responsibility |
| --- | --- |
| `types.py` | `AgentRequest`, `AgentResponse`, `ToolCall`, `Usage`: the provider-independent data exchanged with an agent. |
| `adapters.py` | The `AgentAdapter` protocol and `FunctionAdapter`. |
| `cases.py` | Case and suite models; strict JSON/YAML loading and validation. |
| `runner.py` | Runs each case on a worker thread with a timeout, then applies the metrics. |
| `metrics.py` | The `Metric` protocol and the four built-in metrics. |
| `results.py` | Per-case results and the aggregate summary. |
| `report.py` | Text and JSON rendering. |
| `loader.py` | Resolves an `--agent` specification to an adapter. |
| `cli.py` | The `david-evals` command. |
| `mock.py` | Deterministic mock agents. |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers data flow, failure
handling, trust boundaries and the design decisions behind this layout.

## Limitations

- **Deterministic checks only.** Output is checked by equality, substring and
  regular expression. There is no semantic similarity or model-graded scoring,
  so cases must be written so that a correct answer is recognisable by text.
- **Single turn.** A case is one request and one response. Multi-turn
  conversations and simulated users are not supported.
- **One run per case.** Real agents are non-deterministic; this version does
  not repeat cases or report variance, so one run is one sample, not a
  reliability estimate.
- **Tool calls are self-reported.** The framework compares the tool calls the
  adapter reports. It does not observe, sandbox or verify tool execution, or
  check tool results.
- **Sequential execution.** Cases run one at a time. Large suites against
  slow agents will be slow.
- **Blocking adapters cannot be interrupted.** A synchronous adapter that
  exceeds its timeout is reported as a timeout and abandoned; its thread keeps
  running until the call returns or the process exits, and any side effects of
  that call still happen. Async adapters are cancelled.
- **Latency is wall-clock time of the adapter call** on the machine running
  the evaluation, including network time. It is not a benchmark.
- **Cost depends on the adapter.** It is only as accurate as the usage data
  and prices you supply.
- **No provider adapters are included.** You write the adapter for your agent.
- **The JSON results format (`schema_version` 0.1) may change** in later
  releases.
- **Not production-hardened.** The test suite covers the behaviour described
  here on Python 3.11 to 3.14. It has not been exercised against large suites,
  long-running agents, or on Windows.

## Development

```bash
python -m pip install -e ".[dev]"
pytest                  # tests
ruff format --check .   # formatting
ruff check .            # linting
mypy                    # type checking (strict)
```

GitHub Actions runs the same checks on every pull request, with the tests on
Python 3.11, 3.12, 3.13 and 3.14.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and
[SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## License

[Apache License 2.0](LICENSE)
