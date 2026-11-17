"""Shared types for UniProt annotations."""

# Portions of this file were refined with LLM assistance.

from typing import Literal

# Supported annotations returned for a resolved UniProtKB entry.
NameType = Literal["gene", "gene_symbol", "full_name"]

# Rules for combining resolved annotations within a protein group.
GroupStrategy = Literal["first", "all", "majority"]
