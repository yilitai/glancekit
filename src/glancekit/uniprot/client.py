"""Resolve UniProtKB accessions through batched REST API queries.

The resolver follows pagination, retries transient request failures, and
reports accession ambiguity separately from missing annotations.
"""

# Portions of this file were refined with LLM assistance.

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import numpy as np
import requests

from ._exceptions import (
    UniProtAmbiguityError,
    UniProtRequestError,
    UniProtResponseError,
)
from ._logging import notebook_verbose
from ._progress import batch_progress
from ._types import NameType
from ._utils import is_uniprot_id, strip_isoform

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


def _extract_name(entry: dict, name_type: NameType) -> str | None:
    """Extract the requested annotation from a validated UniProtKB record."""
    if name_type in ("gene", "gene_symbol"):
        genes = entry.get("genes", [])
        if not isinstance(genes, list):
            raise UniProtResponseError("Malformed UniProt genes field")
        name = None
        for gene in genes:
            if not isinstance(gene, dict):
                raise UniProtResponseError("Malformed UniProt gene record")
            primary = gene.get("geneName") or {}
            if not isinstance(primary, dict):
                raise UniProtResponseError("Malformed UniProt geneName field")
            if primary.get("value"):
                name = primary["value"]
                break
        if name is None:
            # Use alternative gene names only when no primary name is present.
            for gene in genes:
                for key in ("orderedLocusNames", "orfNames", "synonyms"):
                    values = gene.get(key) or []
                    if not isinstance(values, list):
                        raise UniProtResponseError(f"Malformed UniProt {key} field")
                    if values and isinstance(values[0], dict) and values[0].get("value"):
                        name = values[0]["value"]
                        break
                if name:
                    break
        if name is None:
            return None
        if not isinstance(name, str):
            raise UniProtResponseError("UniProt returned a non-string gene name")
        return name.upper() if name_type == "gene_symbol" else name

    if name_type == "full_name":
        description = entry.get("proteinDescription") or {}
        if not isinstance(description, dict):
            raise UniProtResponseError("Malformed UniProt proteinDescription field")
        recommended = description.get("recommendedName") or {}
        if not isinstance(recommended, dict):
            raise UniProtResponseError("Malformed UniProt recommendedName field")
        full = recommended.get("fullName") or {}
        if not isinstance(full, dict):
            raise UniProtResponseError("Malformed UniProt fullName field")
        name = full.get("value")
        if name is None:
            submitted = description.get("submissionNames") or []
            if not isinstance(submitted, list):
                raise UniProtResponseError("Malformed UniProt submissionNames field")
            if submitted:
                item = submitted[0]
                if not isinstance(item, dict) or not isinstance(item.get("fullName"), dict):
                    raise UniProtResponseError("Malformed UniProt submitted protein name")
                name = item["fullName"].get("value")
        if name is not None and not isinstance(name, str):
            raise UniProtResponseError("UniProt returned a non-string protein name")
        return name
    raise ValueError(f"Unsupported name_type: {name_type!r}")


def _next_link(header: str | None) -> str | None:
    """Extract and validate the next-page URL from a REST Link header."""
    if not header:
        return None
    for part in header.split(","):
        sections = part.strip().split(";")
        if any(section.strip().replace('"', '') == 'rel=next' for section in sections[1:]):
            value = sections[0].strip()
            if value.startswith("<") and value.endswith(">"):
                link = value[1:-1]
                parsed = urlparse(link)
                if parsed.scheme == "https" and parsed.hostname == "rest.uniprot.org" and parsed.path.startswith("/uniprotkb/search"):
                    return link
                raise UniProtResponseError("Unexpected UniProt pagination URL")
    return None


class _BatchResolver:
    """Query exact primary and secondary accessions with bounded network retries."""

    def __init__(
        self,
        *,
        batch_size: int,
        max_workers: int,
        timeout: float,
        retry: int,
        network_retry_delay: float,
        max_network_retries: int,
        verbose: bool,
        progress: bool,
    ) -> None:
        """Validate request limits and retain the execution settings."""
        if batch_size <= 0 or batch_size > 500:
            raise ValueError("batch_size must be in [1, 500]")
        if max_workers <= 0 or timeout <= 0 or retry <= 0:
            raise ValueError("max_workers, timeout and retry must be positive")
        if max_network_retries <= 0 or network_retry_delay < 0:
            raise ValueError("max_network_retries must be positive and network_retry_delay nonnegative")
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.retry = retry
        self.network_retry_delay = network_retry_delay
        self.max_network_retries = max_network_retries
        self.verbose = verbose
        self.progress = progress

    def _request_batch(self, batch: list[str]) -> list[dict]:
        """Retrieve all pages for one batch and validate their record structure."""
        # Include exact matches against both primary and secondary accessions.
        query = " OR ".join(f"(accession:{uid} OR sec_acc:{uid})" for uid in batch)
        params = {"query": query, "format": "json", "size": min(500, len(batch) * 2 + 5)}
        # Full records retain the aliases needed to identify the requested IDs.
        url: str | None = _SEARCH_URL
        collected: list[dict] = []
        visited: set[str] = set()
        while url:
            if url in visited:
                raise UniProtResponseError("UniProt pagination loop detected")
            visited.add(url)
            response = None
            for attempt in range(self.retry):
                try:
                    response = requests.get(
                        url,
                        params=params if url == _SEARCH_URL else None,
                        timeout=self.timeout,
                        headers={"Accept": "application/json", "User-Agent": "glancekit-uniprot"},
                    )
                    response.raise_for_status()
                    break
                except requests.exceptions.HTTPError as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    if status is None:
                        status = getattr(response, "status_code", None)
                    if status is not None and 400 <= status < 500 and status not in (408, 429):
                        raise UniProtResponseError(
                            f"UniProt rejected the query (HTTP {status}) for a batch of "
                            f"{len(batch)} accessions starting at {batch[0]}. "
                            "Check query syntax or reduce batch_size."
                        ) from exc
                    if attempt + 1 == self.retry:
                        raise UniProtRequestError(f"UniProt request failed for {batch}: {exc}") from exc
                    time.sleep(attempt + 1)
                except requests.exceptions.RequestException as exc:
                    if attempt + 1 == self.retry:
                        raise UniProtRequestError(f"UniProt request failed for {batch}: {exc}") from exc
                    time.sleep(attempt + 1)
            try:
                payload = response.json()
            except (ValueError, TypeError) as exc:
                raise UniProtResponseError(f"Invalid JSON for batch {batch}") from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise UniProtResponseError(f"Malformed UniProt response for batch {batch}: missing results list")
            for entry in payload["results"]:
                if not isinstance(entry, dict) or not isinstance(entry.get("primaryAccession"), str):
                    raise UniProtResponseError(f"Malformed UniProt entry in batch {batch}")
                secondary = entry.get("secondaryAccessions", [])
                if not isinstance(secondary, list) or not all(isinstance(item, str) for item in secondary):
                    raise UniProtResponseError("Malformed UniProt secondaryAccessions field")
                collected.append(entry)
            url = _next_link(response.headers.get("Link"))
            params = None
        return collected

    def resolve(self, accessions: list[str]) -> dict[str, list[dict]]:
        """Return all matching entries for each accession; raise on failed batches."""
        matched: dict[str, dict[str, dict]] = {uid: {} for uid in accessions}
        pending = [accessions[i:i + self.batch_size] for i in range(0, len(accessions), self.batch_size)]
        for round_number in range(1, self.max_network_retries + 1):
            if not pending:
                break
            failed: list[tuple[list[str], Exception]] = []
            if self.verbose:
                logger.info("UniProt request round %d: %d batches", round_number, len(pending))
            with batch_progress(len(pending), round_number, self.progress) as bar:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = {pool.submit(self._request_batch, batch): batch for batch in pending}
                    for future in as_completed(futures):
                        batch = futures[future]
                        try:
                            entries = future.result()
                        except UniProtRequestError as exc:
                            failed.append((batch, exc))
                            continue
                        finally:
                            if bar is not None:
                                bar.update(1)
                        # Index each record by its matched input aliases and unique primary ID.
                        requested = set(batch)
                        for entry in entries:
                            primary = entry["primaryAccession"].upper()
                            aliases = {primary, *(value.upper() for value in entry.get("secondaryAccessions", []))}
                            for accession in requested.intersection(aliases):
                                matched[accession][primary] = entry
            if not failed:
                break
            if round_number == self.max_network_retries:
                failed_ids = [uid for batch, _ in failed for uid in batch]
                raise UniProtRequestError(
                    f"UniProt requests failed after {round_number} round(s); "
                    f"network-unreachable accessions: {failed_ids}. "
                    f"Example: {failed[0][1]}"
                )
            pending = [batch for batch, _ in failed]
            time.sleep(self.network_retry_delay)
        return {uid: list(entries.values()) for uid, entries in matched.items()}


def _map_names(
    uniprot_ids: Iterable[str],
    *,
    name_type: NameType = "gene",
    batch_size: int = 10,
    max_workers: int = 8,
    timeout: float = 15,
    retry: int = 3,
    network_retry_delay: float = 5.0,
    max_network_retries: int = 3,
    name_overrides: Mapping[str, str] | None = None,
    verbose: bool = False,
    progress: bool = True,
) -> tuple[dict[str, str | None], dict[str, bool]]:
    """Return names and resolution flags keyed by the original identifiers.

    Isoform suffixes and letter case are normalized for lookup. Overrides
    are validated against names returned by UniProtKB for the requested
    annotation type. A completed query may yield no annotation.
    """
    if name_type not in ("gene", "gene_symbol", "full_name"):
        raise ValueError(f"Invalid name_type: {name_type!r}")
    raw = list(uniprot_ids)
    if not all(isinstance(uid, str) for uid in raw):
        raise ValueError("Every UniProt identifier must be a string")
    cleaned = [strip_isoform(uid).upper() for uid in raw]
    overrides = dict(name_overrides or {})
    if not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in overrides.items()):
        raise ValueError("name_overrides must map accession strings to nonempty name strings")
    override_by_base: dict[str, str] = {}
    for key, value in overrides.items():
        base = strip_isoform(key).upper()
        if not is_uniprot_id(base):
            raise ValueError(f"Invalid accession in name_overrides: {key!r}")
        if base in override_by_base and override_by_base[base] != value:
            raise ValueError(f"Conflicting name_overrides for {base}")
        override_by_base[base] = value
    valid = list(dict.fromkeys(uid for uid in cleaned if is_uniprot_id(uid)))
    unused = set(override_by_base) - set(valid)
    if unused:
        raise ValueError(f"name_overrides contains accessions absent from input: {sorted(unused)}")
    candidates = _BatchResolver(
        batch_size=batch_size, max_workers=max_workers, timeout=timeout, retry=retry,
        network_retry_delay=network_retry_delay, max_network_retries=max_network_retries,
        verbose=verbose, progress=progress,
    ).resolve(valid)
    ambiguity: list[str] = []
    invalid_override: list[str] = []
    chosen: dict[str, str | None] = {}
    for uid in valid:
        entries = candidates[uid]
        annotations = [(entry["primaryAccession"], _extract_name(entry, name_type)) for entry in entries]
        # Ambiguity is determined by distinct primary accessions.
        if uid in override_by_base:
            selected = override_by_base[uid]
            if selected not in {name for _, name in annotations if name is not None}:
                invalid_override.append(f"{uid}: requested {selected!r}; candidates {annotations}")
            else:
                chosen[uid] = selected
        elif len(entries) > 1:
            ambiguity.append(f"{uid}: {annotations}")
        elif annotations:
            chosen[uid] = annotations[0][1]
        else:
            chosen[uid] = None
    if ambiguity or invalid_override:
        messages = []
        if ambiguity:
            messages.append("Ambiguous UniProt accessions (provide name_overrides):\n  " + "\n  ".join(sorted(ambiguity)))
        if invalid_override:
            messages.append("Invalid name_overrides:\n  " + "\n  ".join(sorted(invalid_override)))
        raise UniProtAmbiguityError("\n".join(messages))
    result = {original: chosen.get(clean) for original, clean in zip(raw, cleaned)}
    flags = {original: chosen.get(clean) is not None for original, clean in zip(raw, cleaned)}
    if verbose:
        logger.info("Resolved %d / %d accessions", sum(flags.values()), len(raw))
        missing = [uid for uid, found in flags.items() if not found]
        if missing:
            logger.info("Unresolved accessions: %s", missing)
    return result, flags


@notebook_verbose
def fetch_uniprot_names(
    uniprot_ids: Iterable[str],
    name_type: NameType = "gene",
    batch_size: int = 10,
    max_workers: int = 8,
    timeout: float = 15,
    retry: int = 3,
    network_retry_delay: float = 5.0,
    max_network_retries: int = 3,
    fallback_to_id: bool = True,
    verbose: bool | None = None,
    progress: bool = True,
    name_overrides: Mapping[str, str] | None = None,
) -> dict[str, str | float]:
    """Resolve UniProtKB accessions to annotation names.

    Parameters
    ----------
    uniprot_ids : iterable of str
        Original UniProtKB accessions, with optional numeric isoform suffixes.
        Strings not resembling accessions remain unresolved.
    name_type : {'gene', 'gene_symbol', 'full_name'}, default='gene'
        Primary gene name, uppercase gene name, or full protein name.
    batch_size : int, default=10
        Maximum accessions per query (up to 500).
    max_workers : int, default=8
        Concurrent API requests.
    timeout : float, default=15
        Requests connection/read timeout in seconds; not an overall deadline.
    retry : int, default=3
        Attempts per HTTP request.
    network_retry_delay : float, default=5
        Wait between rounds of failed batches, in seconds.
    max_network_retries : int, default=3
        Total number of rounds, including the first.
    fallback_to_id : bool, default=True
        Return the original accession for a legitimate unresolved query;
        otherwise return ``numpy.nan``.
    verbose : bool or None, default=None
        Emit detailed INFO messages. The default enables messages when
        tqdm is unavailable and suppresses them when tqdm is available.
    progress : bool, default=True
        Display a batch progress bar when tqdm is available. This option
        controls the progress bar independently of ``verbose``.
    name_overrides : mapping of str to str, optional
        Requested accession-to-name choices, validated against actual UniProt
        candidates for the current ``name_type``. Never bypasses API queries.

    Returns
    -------
    dict
        Mapping from each original accession to a name or fallback value.

    Raises
    ------
    UniProtRequestError
        An HTTP request failed after bounded retries; no partial map returned.
    UniProtResponseError
        UniProt returned invalid or structurally unexpected content.
    UniProtAmbiguityError
        A requested accession matched multiple current UniProtKB entries or a
        name override does not match any candidate. All candidates are listed.
    ValueError
        Invalid arguments or override keys/values.

    Notes
    -----
    Network errors are never treated as legitimate missing records.
    """
    names, _ = _map_names(
        uniprot_ids, name_type=name_type, batch_size=batch_size,
        max_workers=max_workers, timeout=timeout, retry=retry,
        network_retry_delay=network_retry_delay,
        max_network_retries=max_network_retries,
        name_overrides=name_overrides, verbose=verbose, progress=progress,
    )
    return {uid: value if value is not None else (uid if fallback_to_id else np.nan)
            for uid, value in names.items()}
