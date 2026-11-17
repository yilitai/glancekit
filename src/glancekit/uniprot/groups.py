"""Resolve protein groups and summarize agreement among member annotations."""

# Portions of this file were refined with LLM assistance.

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping

from ._logging import notebook_verbose
from ._types import GroupStrategy, NameType
from ._utils import dedupe_preserve_order, strip_isoform
from .client import _map_names

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _merge_group_names(
    resolved_members: list[tuple[str, str]],
    strategy: GroupStrategy,
    name_join: str,
) -> str:
    """Combine resolved names according to the requested group strategy."""
    names = [name for _, name in resolved_members]
    if strategy == "first":
        return names[0]
    if strategy == "all":
        return name_join.join(dedupe_preserve_order(names))
    if strategy == "majority":
        # First occurrence resolves ties between equally frequent names.
        counts = Counter(names)
        return max(dedupe_preserve_order(names), key=lambda name: counts[name])
    raise ValueError(f"Unknown group_strategy: {strategy!r}")


@notebook_verbose
def fetch_uniprot_names_grouped(
    uniprot_ids: Iterable[str],
    name_type: NameType = "gene",
    group_sep: str = ";",
    group_strategy: GroupStrategy = "first",
    name_join: str = ";",
    batch_size: int = 10,
    max_workers: int = 8,
    timeout: float = 15,
    retry: int = 3,
    network_retry_delay: float = 5.0,
    max_network_retries: int = 3,
    verbose: bool | None = None,
    progress: bool = True,
    return_diagnostics: bool = False,
    name_overrides: Mapping[str, str] | None = None,
):
    """Resolve protein groups defined by a configurable accession separator.

    Parameters
    ----------
    uniprot_ids : iterable of str
        Group identifiers containing one or more nonempty member identifiers.
    name_type : {'gene', 'gene_symbol', 'full_name'}, default='gene'
        UniProt primary gene name, uppercase gene name, or protein full name.
    group_sep : str, default=';'
        Literal separator between accessions in each protein group.
    group_strategy : {'first', 'all', 'majority'}, default='first'
        ``first`` selects the first resolved member; ``all`` joins distinct
        resolved names; ``majority`` selects the most frequent resolved name,
        breaking ties by first occurrence (relative plurality vote).
    name_join : str, default=';'
        Separator used to join names in the ``all`` strategy.
    batch_size : int, default=10
        Maximum accession count per UniProt request.
    max_workers : int, default=8
        Concurrent API requests.
    timeout : float, default=15
        Individual request timeout in seconds.
    retry : int, default=3
        Attempts per HTTP request.
    network_retry_delay : float, default=5
        Seconds between network retry rounds.
    max_network_retries : int, default=3
        Maximum request rounds including the initial round.
    verbose : bool or None, default=None
        Emit detailed INFO messages. The default enables messages when
        tqdm is unavailable and suppresses them when tqdm is available.
    progress : bool, default=True
        Display a batch progress bar when tqdm is available. This option
        controls the progress bar independently of ``verbose``.
    return_diagnostics : bool, default=False
        Also return per-group counts, distinct names, and ambiguity flags.
    name_overrides : mapping of str to str, optional
        Individual UniProt accessions to names validated against API results.

    Returns
    -------
    dict or tuple of dict
        Mapping from each original group to its selected annotation, with
        the original group retained if no member resolves. When requested,
        the second dictionary contains ``n_members``, ``n_resolved``,
        ``distinct_genes``, and ``is_ambiguous`` for each group.

    Raises
    ------
    ValueError
        Invalid separator, strategy, or group identifiers.
    UniProtRequestError
        A network request exhausted its retries.
    UniProtResponseError
        An API response was malformed.
    UniProtAmbiguityError
        Secondary accession ambiguity or invalid override.

    Notes
    -----
    ``is_ambiguous`` reflects distinct *resolved* names only. Missing name
    annotations do not prove members refer to the same gene. Every member's
    HTTP query must nevertheless complete successfully.
    """
    if not isinstance(group_sep, str) or not group_sep:
        raise ValueError("group_sep must be a nonempty string")
    if not isinstance(name_join, str):
        raise ValueError("name_join must be a string")
    if group_strategy not in ("first", "all", "majority"):
        raise ValueError(f"Unknown group_strategy: {group_strategy!r}")
    groups = list(uniprot_ids)
    if not all(isinstance(item, str) for item in groups):
        raise ValueError("Every protein group identifier must be a string")
    members_by_group: dict[str, list[str]] = {}
    all_members: list[str] = []
    for group in groups:
        parts = group.split(group_sep)
        if any(not part.strip() for part in parts):
            raise ValueError(f"Empty accession in protein group {group!r}")
        members = [strip_isoform(part) for part in parts]
        members_by_group[group] = members
        all_members.extend(members)
    # Query shared accessions once while retaining member order for voting.
    names, flags = _map_names(
        list(dict.fromkeys(all_members)), name_type=name_type,
        batch_size=batch_size, max_workers=max_workers, timeout=timeout,
        retry=retry, network_retry_delay=network_retry_delay,
        max_network_retries=max_network_retries,
        name_overrides=name_overrides, verbose=verbose, progress=progress,
    )
    mapping: dict[str, str] = {}
    diagnostics: dict[str, dict] = {}
    for group in groups:
        members = members_by_group[group]
        resolved = [(uid, names[uid]) for uid in members if flags[uid]]
        distinct = sorted({name for _, name in resolved})
        diagnostics[group] = {
            "n_members": len(members),
            "n_resolved": len(resolved),
            "distinct_genes": distinct,
            "is_ambiguous": len(distinct) > 1,
        }
        mapping[group] = _merge_group_names(resolved, group_strategy, name_join) if resolved else group
    if verbose:
        logger.info("Resolved %d / %d protein groups", sum(d["n_resolved"] > 0 for d in diagnostics.values()), len(groups))
        logger.info("Ambiguous protein groups: %d", sum(d["is_ambiguous"] for d in diagnostics.values()))
    return (mapping, diagnostics) if return_diagnostics else mapping
