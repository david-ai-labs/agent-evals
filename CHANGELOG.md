# Changelog

## 0.1.0 (unreleased)

First release.

- `AgentAdapter` protocol for connecting any sync or async agent.
- Evaluation suites defined in JSON or YAML, with strict validation.
- `EvalRunner` with per-case, per-suite and global timeouts.
- Metrics: task success, tool-call correctness, latency, and cost where the
  agent reports usage data.
- Deterministic mock agents and two example suites that need no API keys.
- A requirement that cannot be checked fails its case: `max_cost_usd` with no
  cost data, or a `matches` search that exceeds its time limit.
- `matches` searches run in a child process with a hard time limit, so a
  pattern with catastrophic backtracking cannot block a run.
- Text report, JSON results, and the `david-evals` command (`run`, `validate`).
