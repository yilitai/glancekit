"""UniProt REST client and single-accession mapping.

A key design invariant is that an unreachable UniProt request is not treated as
biological evidence that an accession is absent. Fast per-request retries and
slower whole-round network retries are therefore handled separately.

Development note
----------------
Portions of this module were created or refined with LLM-assisted code
generation and review. Final integration and validation remain the
responsibility of the package authors.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

from ._types import NameType
from ._utils import is_uniprot_id, strip_isoform

_UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
_UNIPROT_FIELDS = "accession,gene_names,protein_name"
_USER_AGENT = "Python/requests glancekit-uniprot"


class _BatchResolver:
    """Resolve clean UniProt accessions using concurrent batched requests.

    ``retry``/``retry_delay`` handle transient failure of one request. A batch
    that still never receives a response is returned to :meth:`resolve` as a
    network failure and retried in later rounds controlled by
    ``network_retry_delay`` and ``max_network_retries``.
    """

    def __init__(
        self,
        name_type: NameType,
        batch_size: int = 50,
        max_workers: int = 8,
        timeout: int = 15,
        retry: int = 3,
        retry_delay: float = 1.0,
        network_retry_delay: float = 10.0,
        max_network_retries: int | None = 10,
        verbose: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if max_workers <= 0:
            raise ValueError("max_workers must be > 0.")
        if timeout <= 0:
            raise ValueError("timeout must be > 0.")
        if retry <= 0:
            raise ValueError("retry must be > 0.")
        if network_retry_delay < 0:
            raise ValueError("network_retry_delay must be >= 0.")
        if max_network_retries is not None and max_network_retries <= 0:
            raise ValueError("max_network_retries must be > 0 or None.")

        self.name_type = name_type
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout = timeout
        self.retry = retry
        self.retry_delay = retry_delay
        self.network_retry_delay = network_retry_delay
        self.max_network_retries = max_network_retries
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _fetch_one_batch(self, batch: list[str]) -> tuple[dict[str, str], bool]:
        """Query one batch and return ``(resolved, request_succeeded)``."""
        query = " OR ".join(f"accession:{uid}" for uid in batch)
        params = {
            "query": query,
            "fields": _UNIPROT_FIELDS,
            "format": "json",
            "size": len(batch) + 5,
        }
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}

        for attempt in range(self.retry):
            try:
                response = requests.get(
                    _UNIPROT_SEARCH_URL,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                resolved = {
                    entry["primaryAccession"]: name
                    for entry in results
                    if (name := self._extract_name(entry))
                    and entry.get("primaryAccession")
                }
                return resolved, True
            except requests.exceptions.RequestException as exc:
                if attempt < self.retry - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                self._log(
                    "[glancekit.uniprot] WARNING: batch failed after "
                    f"{self.retry} quick attempts because of a network error: {exc}"
                )
                return {}, False

        return {}, False

    def _extract_name(self, entry: dict) -> str | None:
        if self.name_type in ("gene", "gene_symbol"):
            name = self._extract_gene_name(entry)
            if name is not None and self.name_type == "gene_symbol":
                name = name.upper()
            return name
        if self.name_type == "full_name":
            return self._extract_full_name(entry)
        raise ValueError(
            "name_type must be one of 'gene', 'gene_symbol', or 'full_name'."
        )

    @staticmethod
    def _extract_gene_name(entry: dict) -> str | None:
        genes = entry.get("genes", [])
        for gene in genes:
            value = gene.get("geneName", {}).get("value")
            if value:
                return value

        for gene in genes:
            for key in ("orderedLocusNames", "orfNames", "synonyms"):
                values = gene.get(key, [])
                if values:
                    return values[0].get("value")
        return None

    @staticmethod
    def _extract_full_name(entry: dict) -> str | None:
        description = entry.get("proteinDescription", {})
        name = description.get("recommendedName", {}).get("fullName", {}).get("value")
        if name is not None:
            return name
        submitted = description.get("submissionNames", [])
        if submitted:
            return submitted[0].get("fullName", {}).get("value")
        return None

    def resolve(self, accessions: list[str]) -> dict[str, str]:
        """Resolve accessions while retrying network failures independently."""
        total = len(accessions)
        if total == 0:
            return {}

        pending = list(accessions)
        resolved: dict[str, str] = {}
        round_num = 0

        while pending:
            round_num += 1
            batches = [
                pending[i : i + self.batch_size]
                for i in range(0, len(pending), self.batch_size)
            ]
            if round_num == 1:
                self._log(
                    f"[glancekit.uniprot] {total} accessions -> {len(batches)} batches, "
                    f"{self.max_workers} concurrent workers"
                )
            else:
                self._log(
                    f"[glancekit.uniprot] retry round {round_num}: re-querying "
                    f"{len(pending)} accession(s) after network failure"
                )

            completed = 0
            network_failed: list[str] = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._fetch_one_batch, batch): batch
                    for batch in batches
                }
                for future in as_completed(futures):
                    batch = futures[future]
                    batch_resolved, request_succeeded = future.result()
                    resolved.update(batch_resolved)
                    if not request_succeeded:
                        network_failed.extend(batch)
                    completed += len(batch)
                    self._log(
                        f"[glancekit.uniprot] {min(completed, len(pending))} / "
                        f"{len(pending)} queried this round; "
                        f"{len(resolved)} / {total} resolved so far"
                    )

            pending = network_failed
            if not pending:
                break

            if (
                self.max_network_retries is not None
                and round_num >= self.max_network_retries
            ):
                self._log(
                    f"[glancekit.uniprot] WARNING: gave up after {round_num} round(s); "
                    f"{len(pending)} accession(s) remain network-unreachable. "
                    "They are not confirmed absent from UniProt: "
                    f"{pending}"
                )
                break

            self._log(
                f"[glancekit.uniprot] {len(pending)} accession(s) still failed because "
                f"of network errors; sleeping {self.network_retry_delay}s before retrying."
            )
            time.sleep(self.network_retry_delay)

        return resolved


def fetch_uniprot_names(
    uniprot_ids,
    name_type: NameType = "gene",
    batch_size: int = 50,
    max_workers: int = 8,
    timeout: int = 15,
    retry: int = 3,
    network_retry_delay: float = 10.0,
    max_network_retries: int | None = 10,
    fallback_to_id: bool = True,
    verbose: bool = True,
) -> dict:
    """Map standalone UniProt accessions to gene or protein names.

    Network failures are retried separately from UniProt-confirmed missing
    entries. When ``fallback_to_id`` is true, unresolved accessions retain the
    original identifier; otherwise their mapping value is ``numpy.nan``.
    """
    raw_ids = list(uniprot_ids)
    clean_ids = [strip_isoform(uid) for uid in raw_ids]
    accessions = list(dict.fromkeys(uid for uid in clean_ids if is_uniprot_id(uid)))

    n_skipped = sum(not is_uniprot_id(uid) for uid in clean_ids)
    if n_skipped and verbose:
        print(
            f"[glancekit.uniprot] {n_skipped} entrie(s) are not UniProt accessions; "
            "they will be returned unchanged."
        )

    resolver = _BatchResolver(
        name_type=name_type,
        batch_size=batch_size,
        max_workers=max_workers,
        timeout=timeout,
        retry=retry,
        network_retry_delay=network_retry_delay,
        max_network_retries=max_network_retries,
        verbose=verbose,
    )
    resolved = resolver.resolve(accessions)

    mapping: dict = {}
    unresolved: list[str] = []
    for raw, clean in zip(raw_ids, clean_ids):
        if not is_uniprot_id(clean):
            mapping[raw] = raw
            continue
        name = resolved.get(clean)
        if name is None:
            unresolved.append(raw)
            mapping[raw] = raw if fallback_to_id else np.nan
        else:
            mapping[raw] = name

    if unresolved and verbose:
        print(
            f"\n[glancekit.uniprot] WARNING: {len(unresolved)} accession(s) could not "
            "be resolved. This can reflect a merged/deleted accession, an entry without "
            "the requested name field, or persistent network failure.\n"
            "  Unresolved list:"
        )
        for uid in unresolved:
            print(f"    {uid}")

    return mapping
