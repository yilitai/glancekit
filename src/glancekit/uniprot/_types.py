"""Shared typing aliases for UniProt mapping utilities.

Development note
----------------
Portions of this module were created or refined with LLM-assisted code
generation and review. Final integration and validation remain the
responsibility of the package authors.
"""

from typing import Literal

NameType = Literal["gene", "gene_symbol", "full_name"]
GroupStrategy = Literal["first", "all", "majority"]
