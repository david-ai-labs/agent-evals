"""Resolve an ``--agent`` specification to an :class:`AgentAdapter`."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from david_agent_evals.adapters import AgentAdapter


class AgentLoadError(Exception):
    """Raised when an agent specification cannot be resolved."""


def load_agent(spec: str) -> AgentAdapter:
    """Load an adapter from ``package.module:attribute`` or ``path/file.py:attribute``.

    The attribute may be an adapter instance, an adapter class, or a
    zero-argument factory; classes and factories are called to get the adapter.

    Loading imports and executes the target module, so only load code you trust.
    """
    target, separator, attribute = spec.rpartition(":")
    if not separator or not target or not attribute:
        raise AgentLoadError(
            f"invalid agent '{spec}': expected 'package.module:attribute' "
            "or 'path/to/file.py:attribute'"
        )
    module = _import_file(Path(target)) if target.endswith(".py") else _import_module(target)
    try:
        candidate = getattr(module, attribute)
    except AttributeError:
        raise AgentLoadError(f"'{target}' has no attribute '{attribute}'") from None

    if isinstance(candidate, type) or not isinstance(candidate, AgentAdapter):
        if not callable(candidate):
            raise AgentLoadError(
                f"'{spec}' is not an agent adapter: it needs a 'name' and a 'run(request)' method"
            )
        try:
            candidate = candidate()
        except Exception as exc:
            raise AgentLoadError(f"calling '{spec}' failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(candidate, AgentAdapter):
        raise AgentLoadError(
            f"'{spec}' produced {type(candidate).__name__}, which is not an agent adapter: "
            "it needs a 'name' and a 'run(request)' method"
        )
    return candidate


def _import_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise AgentLoadError(f"could not import module '{name}': {exc}") from exc


def _import_file(path: Path) -> ModuleType:
    if not path.is_file():
        raise AgentLoadError(f"agent file not found: {path}")
    resolved = path.resolve()
    module_name = f"_david_agent_evals_agent_{resolved.stem}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise AgentLoadError(f"could not load agent file: {path}")
    module = importlib.util.module_from_spec(spec)
    # Let the file import its siblings, as it could if run as a script.
    parent = str(resolved.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[module_name]
        raise AgentLoadError(f"importing {path} failed: {type(exc).__name__}: {exc}") from exc
    return module
