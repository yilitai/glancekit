"""Optional, widget-free progress reporting for UniProt requests."""

# Portions of this file were refined with LLM assistance.

from __future__ import annotations

from contextlib import contextmanager


def _load_tqdm():
    """Return the text-mode tqdm factory when the optional import succeeds."""
    try:
        from tqdm.std import tqdm
    except ImportError:
        return None
    return tqdm


def resolve_verbose(verbose: bool | None) -> bool:
    """Use text logs by default only when tqdm is unavailable."""
    return _load_tqdm() is None if verbose is None else verbose


@contextmanager
def batch_progress(total: int, round_number: int, enabled: bool):
    """Yield a bar for one round of batches, or None when unavailable."""
    factory = _load_tqdm() if enabled and total else None
    if factory is None:
        yield None
        return
    with factory(
        total=total,
        desc=f"[glancekit] UniProt round {round_number}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    ) as bar:
        yield bar
