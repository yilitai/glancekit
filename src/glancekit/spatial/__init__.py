"""Spatial cohort tables, AnnData construction, and segmentation plots."""

# Portions of this file were refined with LLM assistance.

from typing import TYPE_CHECKING

from .io import convert_cohort, load_spatial_columns, load_spatial_table

if TYPE_CHECKING:
    from .anndata import make_adata
    from .mask import assign_mask_ids
    from .plotting import plot_mask

__all__ = [
    "assign_mask_ids",
    "convert_cohort",
    "load_spatial_columns",
    "load_spatial_table",
    "make_adata",
    "plot_mask",
]


def __getattr__(name: str):
    """Load optional AnnData, mask, and plotting functions on first access."""
    if name == "make_adata":
        from .anndata import make_adata

        return make_adata
    if name == "assign_mask_ids":
        from .mask import assign_mask_ids

        return assign_mask_ids
    if name == "plot_mask":
        from .plotting import plot_mask

        return plot_mask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
