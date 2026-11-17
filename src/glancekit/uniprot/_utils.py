"""Internal validation and annotation helpers."""

# Portions of this file were refined with LLM assistance.

from __future__ import annotations

import re
from collections.abc import Iterable

# Match canonical six- and ten-character UniProtKB accessions.
_ACCESSION_RE = re.compile(
    r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})",
    re.ASCII,
)
_ISOFORM_RE = re.compile(r"-[1-9][0-9]*\Z", re.ASCII)


def strip_isoform(accession: str) -> str:
    """Return an accession without its numeric isoform suffix."""
    if not isinstance(accession, str):
        raise TypeError(f"UniProt accession must be a string, got {type(accession).__name__}")
    value = accession.strip()
    prefix, separator, suffix = value.rpartition("-")
    if separator and _ISOFORM_RE.fullmatch(f"-{suffix}"):
        return prefix
    return value


def is_uniprot_id(value: object) -> bool:
    """Check the complete UniProtKB accession syntax (isoforms accepted)."""
    if not isinstance(value, str):
        return False
    return _ACCESSION_RE.fullmatch(strip_isoform(value).upper()) is not None


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    """Return distinct items in first-occurrence order."""
    return list(dict.fromkeys(items))


def is_resolved(name: object) -> bool:
    """Return whether a resolver produced a nonempty string annotation."""
    return isinstance(name, str) and bool(name.strip())


def deduplicate_names(names: list[str]) -> list[str]:
    """Append numbered suffixes without colliding with original names."""
    # Reserve every original name, including names encountered later.
    reserved = set(names)
    used: set[str] = set()
    counters: dict[str, int] = {}
    output: list[str] = []
    for name in names:
        if name not in used:
            chosen = name
        else:
            number = counters.get(name, 1) + 1
            chosen = f"{name}_{number}"
            while chosen in used or chosen in reserved:
                number += 1
                chosen = f"{name}_{number}"
            counters[name] = number
        used.add(chosen)
        output.append(chosen)
    return output
