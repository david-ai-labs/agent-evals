"""Command-line interface: ``david-evals run`` and ``david-evals validate``."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

import david_agent_evals
from david_agent_evals.cases import CaseValidationError, EvalSuite, load_suite
from david_agent_evals.loader import AgentLoadError, load_agent
from david_agent_evals.report import render_text, to_json, write_json
from david_agent_evals.runner import DEFAULT_TIMEOUT_SECONDS, EvalRunner

EXIT_OK = 0
EXIT_CASES_FAILED = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="david-evals",
        description="Evaluate AI agent reliability against JSON or YAML case files.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {david_agent_evals.__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="run evaluation suites against an agent",
        description="Run evaluation suites against an agent. Exits 0 if every case passed, "
        "1 if any case failed, errored or timed out, and 2 on a usage or configuration error.",
    )
    run.add_argument("suites", nargs="+", metavar="SUITE", help="case file (.json, .yaml, .yml)")
    run.add_argument(
        "--agent",
        required=True,
        metavar="SPEC",
        help="agent adapter as 'package.module:attribute' or 'path/to/file.py:attribute'; "
        "the module is imported and executed, so only load code you trust",
    )
    run.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help="timeout for cases that set none themselves and whose suite sets no default "
        "(default: %(default)g)",
    )
    run.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="TAG",
        help="only run cases with this tag; repeat to select several",
    )
    run.add_argument("--output", metavar="FILE", help="also write the JSON results to FILE")
    run.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="what to print to stdout (default: %(default)s)",
    )

    validate = commands.add_parser("validate", help="check case files without running an agent")
    validate.add_argument("suites", nargs="+", metavar="SUITE", help="case file to check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        suites = [load_suite(path) for path in args.suites]
    except (CaseValidationError, OSError) as exc:
        return _fail(str(exc))
    if args.command == "validate":
        for suite in suites:
            print(f"{suite.source}: OK ({len(suite.cases)} case(s))")
        return EXIT_OK
    return _run(args, suites)


def _run(args: argparse.Namespace, suites: list[EvalSuite]) -> int:
    # Console scripts do not put the working directory on sys.path; add it so
    # '--agent my_package.agents:agent' resolves against the current project.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        adapter = load_agent(args.agent)
    except AgentLoadError as exc:
        return _fail(str(exc))

    if args.tag and not any(suite.select(args.tag) for suite in suites):
        return _fail(f"no cases match tag(s): {', '.join(args.tag)}")

    runner = EvalRunner(adapter, default_timeout_seconds=args.timeout)
    results = [runner.run_suite(suite, tags=args.tag) for suite in suites]

    if args.output:
        try:
            write_json(results, args.output)
        except OSError as exc:
            return _fail(f"could not write {args.output}: {exc}")
    print(to_json(results) if args.format == "json" else render_text(results))
    if args.output and args.format == "text":
        print(f"\nJSON results written to {args.output}")
    return EXIT_OK if all(result.all_passed for result in results) else EXIT_CASES_FAILED


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_USAGE


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
