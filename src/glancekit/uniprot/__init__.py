"""UniProtKB accession and protein-group annotation helpers for glancekit."""

# Portions of this file were refined with LLM assistance.

from ._exceptions import (
    UniProtAmbiguityError,
    UniProtMappingError,
    UniProtRequestError,
    UniProtResponseError,
)
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
    "UniProtMappingError",
    "UniProtRequestError",
    "UniProtResponseError",
    "UniProtAmbiguityError",
]
