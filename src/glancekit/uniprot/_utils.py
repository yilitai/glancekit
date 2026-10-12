"""Internal identifier and name helpers for UniProt mapping.

Development note
----------------
Portions of this module were created or refined with LLM-assisted code
generation and review. Final integration and validation remain the
responsibility of the package authors.
"""

from __future__ import annotations

import math
import re

_UNIPROT_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]"
    r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"
    r"|^[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]",
    re.IGNORECASE,
)


def is_uniprot_id(value: str) -> bool:
    """Return whether ``value`` resembles a UniProt accession.

    Isoform suffixes such as ``-2`` are ignored for validation, matching the
    behavior of the original mapper.
    """
    base = value.split("-")[0].strip()
    return bool(_UNIPROT_RE.match(base))


def strip_isoform(accession: str) -> str:
    """Return the base accession before an optional isoform suffix."""
    return accession.split("-")[0].strip()


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate strings while preserving first-occurrence order."""
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def is_resolved(name) -> bool:
    """Distinguish NaN/None unresolved values from resolved strings."""
    if name is None:
        return False
    if isinstance(name, float):
        return not math.isnan(name)
    return True


def deduplicate_names(names: list[str]) -> list[str]:
    """Make names unique by appending ``_2``, ``_3``, ... to duplicates."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            unique.append(name)
        else:
            seen[name] += 1
            unique.append(f"{name}_{seen[name]}")
    return unique
