"""Polars-backed CSV/Parquet I/O for multi-image spatial proteomics cohorts.

Lazy scans support column projection and image-level filtering before
materialization. Frequently accessed cohorts can be cached as partitioned
Parquet datasets.
"""

# Portions of this file were refined with LLM assistance.

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import polars as pl


_CACHE_VERSION = 1
_MANIFEST = "_glancekit_cohort.json"
PathLike = str | Path
ReturnType = Literal["pandas", "polars"]


def _file(path: PathLike) -> Path:
    """Resolve a path and validate its existence and table format."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Cohort file not found: {source}")
    if source.suffix.lower() not in {".csv", ".parquet"}:
        raise ValueError(f"Expected a CSV or Parquet file: {source}")
    return source


def _dataset(path: Path) -> bool:
    """Identify a directory containing a glancekit cohort manifest."""
    return path.is_dir() and (path / _MANIFEST).is_file()


def _scan(path: Path, separator: str, infer_schema_length: int) -> pl.LazyFrame:
    """Open a lazy scan for a source table or a converted cohort."""
    if _dataset(path):
        return pl.scan_parquet(path / "data", hive_partitioning=True)
    if path.is_dir():
        raise ValueError(f"Not a converted glancekit cohort: {path}")
    source = _file(path)
    if source.suffix.lower() == ".parquet":
        return pl.scan_parquet(source)
    return pl.scan_csv(source, separator=separator,
                       infer_schema_length=infer_schema_length)


def _names(frame: pl.LazyFrame) -> list[str]:
    """Read the column names from a lazy scan's schema."""
    return frame.collect_schema().names()


def _check(names: Sequence[str], available: Sequence[str], context: str) -> None:
    """Raise a contextual error for any requested columns that are absent."""
    missing = [name for name in names if name not in available]
    if missing:
        raise KeyError(f"Unknown {context}: {missing}; available: {list(available)}")


def _group_cols(group_col: str | Sequence[str]) -> list[str]:
    """Normalize and validate the names of the partition columns."""
    names = [group_col] if isinstance(group_col, str) else list(group_col)
    if not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("group_col must contain one or more nonempty column names.")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate group_col names.")
    return names


def _metadata(source: Path, group_cols: Sequence[str], separator: str,
              infer_schema_length: int) -> dict[str, Any]:
    """Record source identity and conversion settings in the cache manifest."""
    stat = source.stat()
    return {
        "version": _CACHE_VERSION,
        "source": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "group_cols": list(group_cols),
        "separator": separator,
        "infer_schema_length": infer_schema_length,
    }


def _check_cache(path: Path, expected: Mapping[str, Any]) -> None:
    """Verify that a cache matches the source and contains Parquet data."""
    if not _dataset(path):
        raise FileExistsError(f"Cache location is not a valid cohort: {path}")
    try:
        with (path / _MANIFEST).open(encoding="utf-8") as handle:
            actual = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid cache manifest: {path}") from exc
    if actual != dict(expected):
        raise ValueError(
            f"Stale or incompatible cohort cache: {path}. "
            "Use another cache_dir or manually remove the outdated cache."
        )
    if not any((path / "data").rglob("*.parquet")):
        raise ValueError(f"No Parquet data found in cache: {path}")


def load_spatial_columns(
    file_path: PathLike,
    *,
    separator: str = ",",
    infer_schema_length: int = 1000,
) -> list[str]:
    """Return column names without collecting data rows.

    Parameters
    ----------
    file_path : str or pathlib.Path
        CSV or Parquet file, or a directory produced by ``convert_cohort``.
    separator : str, default=','
        Field delimiter for CSV input.
    infer_schema_length : int, default=1000
        Number of CSV rows used by Polars to infer data types.

    Returns
    -------
    list of str
        Column names in scan order.
    """
    return _names(_scan(Path(file_path), separator, infer_schema_length))


def convert_cohort(
    file_path: PathLike,
    *,
    output_dir: PathLike,
    group_col: str | Sequence[str],
    separator: str = ",",
    infer_schema_length: int = 1000,
) -> Path:
    """Stream a CSV/Parquet cohort into a partitioned Parquet dataset.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Source CSV or Parquet file.
    output_dir : str or pathlib.Path
        Destination directory for the partitioned data and manifest.
    group_col : str or sequence of str
        Partition columns, such as ``("sample_id", "roi_id")``.
    separator : str, default=','
        Field delimiter for CSV input.
    infer_schema_length : int, default=1000
        Number of CSV rows used by Polars to infer data types.

    Returns
    -------
    pathlib.Path
        Absolute path to the published cohort directory.

    Raises
    ------
    FileExistsError
        The destination already exists.
    KeyError
        A partition column is absent from the source table.
    ValueError
        The source format or partition specification is invalid, or the
        conversion produces no data.

    Notes
    -----
    The output is staged in a temporary directory next to the destination
    and published after all data and metadata have been written.
    """
    source = _file(file_path)
    keys = _group_cols(group_col)
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    frame = _scan(source, separator, infer_schema_length)
    _check(keys, _names(frame), "partition columns")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        data_dir = temporary / "data"
        frame.sink_parquet(
            pl.PartitionBy(data_dir, key=keys, include_key=False),
            mkdir=True,
        )
        if not any(data_dir.rglob("*.parquet")):
            raise ValueError("No data written; cohort may be empty.")
        metadata = _metadata(source, keys, separator, infer_schema_length)
        with (temporary / _MANIFEST).open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        if target.exists():
            raise FileExistsError(f"Output was created concurrently: {target}")
        temporary.rename(target)
        return target
    finally:
        # Clean up only the unique temporary directory created above.
        if temporary.exists():
            shutil.rmtree(temporary)


def _cache_path(source: Path, cache_dir: PathLike | None) -> Path:
    """Derive a stable cache path from the absolute source path."""
    root = Path.cwd() / "glancekit_cache" if cache_dir is None else Path(cache_dir)
    digest = hashlib.sha256(str(source).encode()).hexdigest()[:12]
    return root.expanduser().resolve() / f"{source.stem}_{digest}"


def load_spatial_table(
    file_path: PathLike,
    *,
    group_col: str | None = None,
    groups: Any = None,
    filters: Mapping[str, Any] | None = None,
    columns: Sequence[str] | None = None,
    drop_columns: Sequence[str] | None = None,
    index_col: str | None = None,
    cache: bool = False,
    cache_dir: PathLike | None = None,
    return_type: ReturnType = "pandas",
    separator: str = ",",
    infer_schema_length: int = 1000,
) -> Any:
    """Select rows and columns from CSV/Parquet spatial cohort data.

    Parameters
    ----------
    file_path : str or pathlib.Path
        CSV or Parquet file, or a directory produced by ``convert_cohort``.
    group_col : str or None, default=None
        Column identifying the image or ROI. Required when using ``groups``
        or creating a cache.
    groups : scalar, collection, or None, default=None
        One or more exact values to select from ``group_col``.
    filters : mapping or None, default=None
        Additional exact-match conditions keyed by column name. Each value
        is a scalar or a list, tuple, set, or frozenset of allowed values.
        Conditions are combined with logical AND.
    columns : sequence of str or None, default=None
        Columns to retain, in output order.
    drop_columns : sequence of str or None, default=None
        Columns to exclude. Mutually exclusive with ``columns``.
    index_col : str or None, default=None
        Column to use as the pandas index. The column is also retained.
    cache : bool, default=False
        Create or reuse a partitioned Parquet cache of the entire source
        file. The source and conversion settings must match the manifest.
    cache_dir : str, pathlib.Path, or None, default=None
        Cache parent directory. The default is ``glancekit_cache`` under the
        current working directory.
    return_type : {'pandas', 'polars'}, default='pandas'
        Output table type. The pandas result requires pandas and PyArrow.
    separator : str, default=','
        Field delimiter for CSV input.
    infer_schema_length : int, default=1000
        Number of CSV rows used by Polars to infer data types.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        Materialized table after applying row filters and column selection.

    Notes
    -----
    Filter columns may be omitted from the output. For composite image IDs,
    specify conditions on each component through ``filters``.
    """
    if return_type not in {"pandas", "polars"}:
        raise ValueError("return_type must be 'pandas' or 'polars'.")
    if groups is not None and group_col is None:
        raise ValueError("groups requires group_col.")
    if columns is not None and drop_columns is not None:
        raise ValueError("columns and drop_columns are mutually exclusive.")
    if cache and not group_col:
        raise ValueError("cache=True requires group_col.")
    if index_col is not None and return_type != "pandas":
        raise ValueError("index_col requires return_type='pandas'.")

    path = Path(file_path).expanduser().resolve()
    if cache:
        if path.is_dir():
            raise ValueError("Already a converted cohort; use cache=False.")
        source = _file(path)
        target = _cache_path(source, cache_dir)
        expected = _metadata(source, [group_col], separator, infer_schema_length)
        if target.exists():
            _check_cache(target, expected)
        else:
            try:
                convert_cohort(source, output_dir=target, group_col=group_col,
                               separator=separator,
                               infer_schema_length=infer_schema_length)
            except FileExistsError:
                _check_cache(target, expected)
        path = target

    frame = _scan(path, separator, infer_schema_length)
    available = _names(frame)
    conditions = dict(filters or {})
    if groups is not None:
        if group_col in conditions:
            raise ValueError(f"Duplicate filter for {group_col!r}.")
        conditions[group_col] = groups
    _check(list(conditions), available, "filter columns")
    for name, value in conditions.items():
        if isinstance(value, (list, tuple, set, frozenset)):
            frame = frame.filter(pl.col(name).is_in(list(value)))
        else:
            frame = frame.filter(pl.col(name) == value)

    if columns is not None:
        selected = list(columns)
        _check(selected, available, "selected columns")
    elif drop_columns is not None:
        _check(list(drop_columns), available, "dropped columns")
        excluded = set(drop_columns)
        selected = [name for name in available if name not in excluded]
    else:
        selected = available
    if index_col is not None and index_col not in selected:
        raise ValueError(f"index_col {index_col!r} must be in output columns.")
    if not selected:
        raise ValueError("Cannot select zero columns.")
    result = frame.select(selected).collect()
    if return_type == "polars":
        return result
    try:
        out = result.to_pandas()
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Pandas output requires pandas and pyarrow; install both, or "
            "request return_type='polars'."
        ) from exc
    return out.set_index(index_col, drop=False) if index_col is not None else out
