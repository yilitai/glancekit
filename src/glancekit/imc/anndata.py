"""AnnData construction helpers for cell-level IMC tables.

Development note
----------------
Portions of this module were created or refined with LLM-assisted code
generation and review. Final integration and validation remain the
responsibility of the package authors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def make_adata(
    df: pd.DataFrame,
    spatial_cols: Sequence[str],
    marker_cols: Sequence[str],
    obs_cols: Sequence[str] | None,
    index_cols: str | None,
) -> "ad.AnnData":
    """Construct an :class:`anndata.AnnData` object from a cell-level IMC table.

    Marker measurements are stored in ``adata.X`` and the two-dimensional
    coordinates specified by ``spatial_cols`` are stored in
    ``adata.obsm['spatial']``.
    """
    import anndata as ad

    spatial_cols = list(spatial_cols)
    marker_cols = list(marker_cols)
    if len(spatial_cols) < 2:
        raise ValueError("spatial_cols must contain at least two coordinate columns.")

    if obs_cols is not None:
        obs_df = df[list(obs_cols)].copy()
        if index_cols is not None:
            if index_cols not in obs_df.columns:
                raise KeyError(
                    f"index_cols {index_cols!r} must also be present in obs_cols."
                )
            obs_df = obs_df.set_index(index_cols, drop=False)
        obs_df.index = obs_df.index.astype(str)
        adata = ad.AnnData(
            X=df[marker_cols].to_numpy(),
            obs=obs_df,
            var=pd.DataFrame(index=marker_cols),
        )
    else:
        adata = ad.AnnData(
            X=df[marker_cols].to_numpy(),
            var=pd.DataFrame(index=marker_cols),
        )

    adata.obsm["spatial"] = df[spatial_cols].to_numpy()

    if index_cols is not None:
        if index_cols not in adata.obs.columns:
            raise KeyError(
                f"index_cols {index_cols!r} is unavailable in adata.obs; provide obs_cols containing it."
            )
        adata.obs_names = adata.obs[index_cols].astype(str)

    return adata
