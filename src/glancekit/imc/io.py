"""Tabular I/O helpers for imaging mass cytometry (IMC) data.

These functions are intentionally format-light: they operate on CSV-like
single-cell tables and do not assume a particular downstream GLANCE model.

Development note
----------------
Portions of this module were created or refined with LLM-assisted code
generation and review. Final integration and validation remain the
responsibility of the package authors.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

PathLike = str | Path


def load_imc_columns(file_path: PathLike) -> pd.Index:
    """Return column names from an IMC CSV file without loading data rows."""
    return pd.read_csv(file_path, nrows=0).columns


def load_imc_table(
    file_path: PathLike,
    target_col: str | None = None,
    target_value: str | None = None,
    chunksize: int = 100000,
    usecols: Sequence[str] | None = None,
    dropcols: Sequence[str] | None = None,
    indexcol: str | None = None,
) -> pd.DataFrame:
    """Load an IMC table in chunks, optionally filtering rows while reading.

    Parameters
    ----------
    file_path
        CSV file containing one row per cell.
    target_col, target_value
        When both are provided, retain only rows where
        ``chunk[target_col] == target_value``. Supplying only one of the pair
        leaves rows unfiltered, matching the historical ``glancekit`` behavior.
    chunksize
        Number of rows read per pandas chunk.
    usecols
        Columns to read. Takes precedence over ``dropcols`` when non-empty.
    dropcols
        Columns to omit when ``usecols`` is not supplied.
    indexcol
        Optional column to use as the returned DataFrame index while retaining
        the column itself.
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be > 0.")

    all_columns = pd.read_csv(file_path, nrows=0).columns.tolist()
    if usecols:
        actual_usecols = list(usecols)
    elif dropcols:
        drop_set = set(dropcols)
        actual_usecols = [column for column in all_columns if column not in drop_set]
    else:
        actual_usecols = all_columns

    if target_col is not None and target_value is not None and target_col not in actual_usecols:
        raise ValueError(
            f"target_col {target_col!r} must be included in the columns being read."
        )

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(file_path, usecols=actual_usecols, chunksize=chunksize):
        if target_col is not None and target_value is not None:
            chunk = chunk.loc[chunk[target_col] == target_value]
        if not chunk.empty:
            chunks.append(chunk)

    if chunks:
        result = pd.concat(chunks, axis=0)
    else:
        result = pd.DataFrame(columns=actual_usecols)

    if indexcol is None:
        return result.reset_index(drop=True)

    if indexcol not in result.columns:
        raise KeyError(f"indexcol {indexcol!r} is not present in the loaded table.")
    return result.set_index(indexcol, drop=False)


def load_imc_id_mapping_count(
    file_path: PathLike,
    id_col: str,
    count_col: str,
    chunksize: int = 100000,
):
    """Collect unique ``count_col`` values per ``id_col`` from a large IMC CSV.

    Returns
    -------
    mapping
        ``defaultdict(set)`` mapping each ID to its unique observed values.
    counts
        Plain dict mapping each ID to the number of unique observed values.
    """
    if chunksize <= 0:
        raise ValueError("chunksize must be > 0.")

    mapping = defaultdict(set)
    for chunk in pd.read_csv(
        file_path,
        usecols=[id_col, count_col],
        chunksize=chunksize,
    ):
        for id_value, count_value in zip(chunk[id_col], chunk[count_col]):
            mapping[id_value].add(count_value)

    counts = {key: len(values) for key, values in mapping.items()}
    return mapping, counts
