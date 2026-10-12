"""Imaging mass cytometry utilities for ``glancekit``.

Public imports are re-exported here so existing code such as
``from glancekit.imc import make_adata`` continues to work after the
module was split into focused files.

Development note
----------------
Portions of this module were created or refined with LLM-assisted code
generation and review. Final integration and validation remain the
responsibility of the package authors.
"""

from .anndata import make_adata
from .io import load_imc_columns, load_imc_id_mapping_count, load_imc_table

__all__ = [
    "load_imc_columns",
    "load_imc_table",
    "load_imc_id_mapping_count",
    "make_adata",
]
