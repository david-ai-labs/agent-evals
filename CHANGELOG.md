# Changelog

## 0.1.0 (unreleased)

First release.

- `AgentAdapter` protocol for connecting any sync or async agent.
- Evaluation suites defined in JSON or YAML, with strict validation.
- `EvalRunner` with per-case, per-suite and global timeouts.
- Metrics: task success, tool-call correctness, latency, and cost where the
  agent reports usage data.
- Deterministic mock agents and two example suites that need no API keys.
- Text report, JSON results, and the `david-evals` command (`run`, `validate`).
