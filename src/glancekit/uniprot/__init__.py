"""UniProt identifier utilities for ``glancekit``.

The public API remains compact while the implementation is split by concern:
REST access in :mod:`glancekit.uniprot.client`, protein-group logic in
:mod:`glancekit.uniprot.groups`, and AnnData integration in
:mod:`glancekit.uniprot.anndata`.

Development note
----------------
Portions of this module were created or refined with LLM-assisted code
generation and review. Final integration and validation remain the
responsibility of the package authors.
"""

from ._types import GroupStrategy, NameType
from .anndata import rename_var_names
from .client import fetch_uniprot_names
from .groups import fetch_uniprot_names_grouped

__all__ = [
    "NameType",
    "GroupStrategy",
    "fetch_uniprot_names",
    "fetch_uniprot_names_grouped",
    "rename_var_names",
]
