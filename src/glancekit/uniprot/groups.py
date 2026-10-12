"""Protein-group handling built on the UniProt single-accession client.

Protein groups can collapse several accessions that resolve either to the same
gene or to genuinely different genes sharing peptide evidence. The latter case
is retained explicitly as group ambiguity diagnostics.

Development note
----------------
Portions of this module were created or refined with LLM-assisted code
generation and review. Final integration and validation remain the
responsibility of the package authors.
"""

from __future__ import annotations

from ._types import GroupStrategy, NameType
from ._utils import dedupe_preserve_order, is_resolved, strip_isoform
from .client import fetch_uniprot_names


def _merge_group_names(
    resolved_members: list[tuple[str, str]],
    total_members: int,
    strategy: GroupStrategy,
    name_join: str,
) -> str:
    """Merge names from resolved members according to ``strategy``."""
    names = [name for _, name in resolved_members]
    if strategy == "first":
        return names[0]
    if strategy == "all":
        return name_join.join(dedupe_preserve_order(names))
    if strategy == "majority":
        ratio = len(resolved_members) / total_members
        return (
            name_join.join(dedupe_preserve_order(names))
            if ratio >= 0.5
            else names[0]
        )
    raise ValueError(f"Unknown group_strategy: {strategy!r}")


def fetch_uniprot_names_grouped(
    uniprot_ids,
    name_type: NameType = "gene",
    group_sep: str = ";",
    group_strategy: GroupStrategy = "first",
    name_join: str = ";",
    batch_size: int = 50,
    max_workers: int = 8,
    timeout: int = 15,
    retry: int = 3,
    network_retry_delay: float = 10.0,
    max_network_retries: int | None = 10,
    verbose: bool = True,
    return_diagnostics: bool = False,
):
    """Map protein-group identifiers to gene or protein names.

    ``return_diagnostics=True`` additionally returns, for each original group,
    ``n_members``, ``n_resolved``, ``distinct_genes``, and ``is_ambiguous``.
    Ambiguity means that resolved members point to more than one distinct name;
    it is therefore preserved independently of the chosen merge strategy.
    """
    raw_ids = list(uniprot_ids)

    id_to_members: dict[str, list[str]] = {}
    all_members: list[str] = []
    for group_id in raw_ids:
        members = [
            strip_isoform(part)
            for part in group_id.split(group_sep)
            if part.strip()
        ]
        id_to_members[group_id] = members
        all_members.extend(members)

    unique_members = list(dict.fromkeys(all_members))
    if verbose:
        print(
            f"[glancekit.uniprot] {len(raw_ids)} protein groups "
            f"(sep={group_sep!r}) -> {len(unique_members)} unique accessions"
        )

    member_mapping = fetch_uniprot_names(
        unique_members,
        name_type=name_type,
        batch_size=batch_size,
        max_workers=max_workers,
        timeout=timeout,
        retry=retry,
        network_retry_delay=network_retry_delay,
        max_network_retries=max_network_retries,
        fallback_to_id=False,
        verbose=verbose,
    )

    mapping: dict[str, str] = {}
    diagnostics: dict[str, dict] = {}
    for group_id in raw_ids:
        members = id_to_members[group_id]
        resolved_members = [
            (member, member_mapping.get(member))
            for member in members
            if is_resolved(member_mapping.get(member))
        ]
        distinct_names = sorted({name for _, name in resolved_members})
        diagnostics[group_id] = {
            "n_members": len(members),
            "n_resolved": len(resolved_members),
            "distinct_genes": distinct_names,
            "is_ambiguous": len(distinct_names) > 1,
        }

        if not resolved_members:
            mapping[group_id] = group_id
        else:
            mapping[group_id] = _merge_group_names(
                resolved_members,
                len(members),
                group_strategy,
                name_join,
            )

    if verbose:
        n_resolved = sum(mapping[group_id] != group_id for group_id in raw_ids)
        n_ambiguous = sum(d["is_ambiguous"] for d in diagnostics.values())
        print(
            f"\n[glancekit.uniprot] Done: {n_resolved} / {len(raw_ids)} groups resolved."
        )
        if n_ambiguous:
            print(
                f"[glancekit.uniprot] WARNING: {n_ambiguous} / {len(raw_ids)} groups "
                "resolved to more than one distinct gene/name. Use diagnostics or "
                "the AnnData ambiguity columns before gene-level interpretation."
            )

    if return_diagnostics:
        return mapping, diagnostics
    return mapping
