"""Notebook-friendly INFO output without modifying the root logger."""

# Portions of this file were refined with LLM assistance.

from __future__ import annotations

import functools
import inspect
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar

from ._progress import resolve_verbose

_ACTIVE = ContextVar("glancekit_uniprot_verbose_active", default=False)


@contextmanager
def _verbose_output(enabled: bool):
    """Temporarily route UniProt package logs to standard error."""
    if not enabled or _ACTIVE.get():
        yield
        return
    package_logger = logging.getLogger(__package__)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[glancekit] %(levelname)s: %(message)s"))
    handler.setLevel(logging.INFO)
    old_level, old_propagate = package_logger.level, package_logger.propagate
    token = _ACTIVE.set(True)
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)
    package_logger.propagate = False
    try:
        yield
    finally:
        package_logger.removeHandler(handler)
        package_logger.setLevel(old_level)
        package_logger.propagate = old_propagate
        _ACTIVE.reset(token)
        handler.close()


def notebook_verbose(func):
    """Resolve automatic verbosity and configure logging for one API call.

    The wrapped function must expose a ``verbose`` parameter. Nested API
    calls receive the resolved value and reuse the active package handler.
    """
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Forward a concrete verbosity value to the public entry point."""
        call = signature.bind(*args, **kwargs)
        enabled = resolve_verbose(call.arguments.get("verbose"))
        call.arguments["verbose"] = enabled
        with _verbose_output(enabled):
            return func(*call.args, **call.kwargs)

    return wrapper
