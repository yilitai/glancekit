"""AnnData integration for UniProt identifier mapping.

Development note
----------------
Portions of this module were created or refined with LLM-assisted code
generation and review. Final integration and validation remain the
responsibility of the package authors.
"""

from __future__ import annotations

from ._types import NameType
from ._utils import deduplicate_names, is_uniprot_id
from .client import fetch_uniprot_names
from .groups import fetch_uniprot_names_grouped


def rename_var_names(
    adata,
    name_type: NameType = "gene",
    grouped: bool = False,
    group_sep: str = ";",
    store_original_key: str | None = "uniprot_id",
    deduplicate: bool = True,
    **kwargs,
) -> dict:
    """Resolve ``adata.var_names`` from UniProt IDs and update them in place.

    In grouped mode, two provenance columns are added to ``adata.var``:
    ``is_ambiguous_group`` and ``distinct_genes_in_group``. The original IDs can
    also be retained in ``adata.var[store_original_key]``.

    Notes
    -----
    Network retries happen inside a single mapping call. Re-running this
    function on an already-renamed object is not a safe retry strategy because
    the original protein-group structure has already been replaced.
    """
    ids = list(adata.var_names)

    if store_original_key is not None:
        adata.var[store_original_key] = ids

    if grouped:
        mapping, diagnostics = fetch_uniprot_names_grouped(
            ids,
            name_type=name_type,
            group_sep=group_sep,
            return_diagnostics=True,
            **kwargs,
        )
        adata.var["is_ambiguous_group"] = [
            diagnostics[identifier]["is_ambiguous"] for identifier in ids
        ]
        adata.var["distinct_genes_in_group"] = [
            ";".join(diagnostics[identifier]["distinct_genes"])
            for identifier in ids
        ]
    else:
        mapping = fetch_uniprot_names(ids, name_type=name_type, **kwargs)

    new_names = [mapping.get(identifier, identifier) for identifier in ids]
    if deduplicate:
        new_names = deduplicate_names(new_names)

    if grouped:
        adata.var["resolved"] = [
            new != original for original, new in zip(ids, new_names)
        ]
    else:
        adata.var["resolved"] = [
            not (is_uniprot_id(original) and new == original)
            for original, new in zip(ids, new_names)
        ]

    adata.var_names = new_names

    n_resolved = int(adata.var["resolved"].sum())
    n_total = len(ids)
    print(
        f"\n[glancekit.uniprot] var_names updated: {n_resolved} / {n_total} resolved.\n"
        "  Filter resolved only:\n"
        "    adata[:, adata.var['resolved']].copy()"
    )

    if grouped:
        n_ambiguous = int(adata.var["is_ambiguous_group"].sum())
        if n_ambiguous:
            print(
                f"  {n_ambiguous} / {n_total} rows are ambiguous protein groups:\n"
                "    adata[:, ~adata.var['is_ambiguous_group']].copy()"
            )

    if n_total - n_resolved > 0:
        unresolved = [
            original
            for original, resolved in zip(ids, adata.var["resolved"])
            if not resolved
        ]
        print(f"\n  Unresolved var_names ({len(unresolved)}):")
        for name in unresolved:
            print(f"    {name}")

    return mapping
