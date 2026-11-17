"""Apply validated UniProt annotations to AnnData variable metadata."""

# Portions of this file were refined with LLM assistance.

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ._logging import notebook_verbose
from ._types import NameType
from ._utils import deduplicate_names
from .client import _map_names
from .groups import fetch_uniprot_names_grouped

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@notebook_verbose
def rename_var_names(
    adata: AnnData,
    name_type: NameType = "gene",
    grouped: bool = False,
    group_sep: str = ";",
    store_original_key: str | None = "uniprot_id",
    deduplicate: bool = True,
    *,
    name_overrides: Mapping[str, str] | None = None,
    verbose: bool | None = None,
    progress: bool = True,
    **kwargs,
) -> dict[str, str]:
    """Rename ``adata.var_names`` using UniProtKB annotations, in place.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData with UniProt identifiers in ``var['uniprot_id']`` (preferred)
        or, if absent, in ``var_names``.
    name_type : {'gene', 'gene_symbol', 'full_name'}, default='gene'
        Gene name, uppercase gene symbol, or protein full name.
    grouped : bool, default=False
        Interpret each source identifier as a protein group separated by
        ``group_sep``. Does not aggregate any abundance values.
    group_sep : str, default=';'
        Literal protein-group accession separator.
    store_original_key : str or None, default='uniprot_id'
        Where to store source IDs on the first successful mapping; ``None``
        disables storing but does not disable reading an existing
        ``var['uniprot_id']`` column.
    deduplicate : bool, default=True
        Generate collision-free unique variable names with numeric suffixes.
    name_overrides : mapping of str to str, optional
        Validated individual accession-to-name choices.
    verbose : bool or None, default=None
        Emit detailed INFO messages. The default enables messages when
        tqdm is unavailable and suppresses them when tqdm is available.
    progress : bool, default=True
        Display a batch progress bar when tqdm is available. This option
        controls the progress bar independently of ``verbose``.
    **kwargs
        Forwarded query options: ``group_strategy`` and ``name_join`` when
        grouped, and ``batch_size``, ``max_workers``, ``timeout``, ``retry``,
        ``network_retry_delay``, and ``max_network_retries`` for either mode.

    Returns
    -------
    dict
        Mapping from each original ID or group to its selected annotation.

    Raises
    ------
    ValueError
        Invalid input column, parameters, or override specification.
    UniProtRequestError
        Network requests exhausted bounded retries.
    UniProtResponseError
        Unexpected API response data.
    UniProtAmbiguityError
        Accession matches multiple current entries without a valid override.

    Notes
    -----
    Repeated calls always prefer the original ``var['uniprot_id']`` when
    present; switching ``name_type`` is safe. Querying and transformation
    occur before any mutation. All known mapping columns are updated or
    removed to match this call. On API or validation errors the original
    AnnData remains unchanged.
    """
    if "uniprot_id" in adata.var.columns:
        source = list(adata.var["uniprot_id"])
    else:
        source = list(adata.var_names)
    bad = [(index, value) for index, value in enumerate(source)
           if not isinstance(value, str) or not value.strip()]
    if bad:
        raise ValueError(f"Invalid UniProt identifiers at adata.var rows: {bad[:20]}"
                         + (f" (and {len(bad)-20} more)" if len(bad) > 20 else ""))
    if store_original_key is not None and (not isinstance(store_original_key, str) or not store_original_key):
        raise ValueError("store_original_key must be a nonempty string or None")
    managed = {"resolved", "is_ambiguous_group", "distinct_genes_in_group"}
    if store_original_key in managed:
        raise ValueError("store_original_key conflicts with a managed annotation column")

    if grouped:
        mapping, diagnostic = fetch_uniprot_names_grouped(
            source, name_type=name_type, group_sep=group_sep,
            return_diagnostics=True, name_overrides=name_overrides,
            verbose=verbose, progress=progress, **kwargs,
        )
        resolved = [diagnostic[uid]["n_resolved"] > 0 for uid in source]
        extra = {
            "is_ambiguous_group": [diagnostic[uid]["is_ambiguous"] for uid in source],
            "distinct_genes_in_group": [";".join(diagnostic[uid]["distinct_genes"]) for uid in source],
        }
    else:
        if "group_strategy" in kwargs or "name_join" in kwargs:
            raise ValueError("group_strategy and name_join require grouped=True")
        names, flags = _map_names(
            source, name_type=name_type, name_overrides=name_overrides,
            verbose=verbose, progress=progress, **kwargs,
        )
        mapping = {uid: names[uid] if flags[uid] else uid for uid in source}
        resolved = [flags[uid] for uid in source]
        extra = {}

    names = [mapping[uid] for uid in source]
    if deduplicate:
        names = deduplicate_names(names)

    # Stage the complete var frame, including deletion of stale grouped fields,
    # before committing anything to the supplied AnnData.
    staged = adata.var.copy(deep=True)
    for column in managed:
        if column in staged.columns:
            staged = staged.drop(columns=column)
    if store_original_key is not None and store_original_key not in staged.columns:
        staged[store_original_key] = source
    staged["resolved"] = resolved
    for column, values in extra.items():
        staged[column] = values
    staged.index = names

    # AnnData var.index is var_names. A single assignment commits both the
    # names and annotation columns, with no partially written intermediate.
    adata.var = staged
    if verbose:
        logger.info("Updated %d / %d AnnData variable names", sum(resolved), len(source))
    return mapping
