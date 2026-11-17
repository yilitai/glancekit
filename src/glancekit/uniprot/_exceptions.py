"""Exceptions for incomplete, invalid, or ambiguous UniProtKB mappings."""

# Portions of this file were refined with LLM assistance.


class UniProtMappingError(RuntimeError):
    """Base exception for an incomplete or invalid mapping operation."""


class UniProtRequestError(UniProtMappingError):
    """Raised when an API request cannot be completed reliably."""


class UniProtResponseError(UniProtMappingError):
    """Raised when an API response cannot be interpreted safely."""


class UniProtAmbiguityError(UniProtMappingError):
    """Raised for ambiguous accessions without valid user disambiguation."""
