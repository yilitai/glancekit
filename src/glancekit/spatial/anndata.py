"""Construct AnnData from one row per cell."""

# Portions of this file were refined with LLM assistance.

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

if TYPE_CHECKING:
    from anndata import AnnData


def make_adata(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    spatial_cols: tuple[str, str],
    obs_cols: Sequence[str] | None = None,
    index_cols: str | Sequence[str] | None = None,
) -> AnnData:
    """Build a cell-by-feature AnnData from a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Source table with one observation per row and unique column names.
    feature_cols : sequence of str
        Real numeric feature columns, in the desired order of ``X`` and
        ``var_names``.
    spatial_cols : tuple of str
        Real numeric x and y coordinate columns, in that order. Coordinate
        units are preserved.
    obs_cols : sequence of str or None, default=None
        Columns copied to ``obs``. The default retains all non-feature
        columns, including coordinate and identifier columns.
    index_cols : str, sequence of str, or None, default=None
        Source columns used to construct ``obs_names``. The default uses
        the input index. Composite IDs are serialized as JSON arrays of
        strings. The resulting IDs must be nonmissing and unique.

    Returns
    -------
    anndata.AnnData
        Observations in input row order with ``X`` and ``obsm["spatial"]``
        stored as float64 arrays. Missing numeric values are represented by
        NaN.

    Raises
    ------
    KeyError
        A requested column is absent from the source table.
    TypeError
        The input is not a pandas DataFrame or a requested numeric column
        has an unsupported dtype.
    ValueError
        Column selections are invalid or the cell IDs are missing or
        nonunique after serialization.
    """
    from anndata import AnnData

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not df.columns.is_unique:
        raise ValueError("Input table contains duplicate column names.")
    if any(isinstance(cols, str) for cols in (feature_cols, spatial_cols, obs_cols)):
        raise TypeError("feature_cols, spatial_cols and obs_cols take sequences of names.")
    features = list(feature_cols)
    coordinates = list(spatial_cols)
    keys = [index_cols] if isinstance(index_cols, str) else list(
        () if index_cols is None else index_cols
    )
    metadata = (
        [col for col in df.columns if col not in features]
        if obs_cols is None else list(obs_cols)
    )
    if not features:
        raise ValueError("feature_cols must contain at least one feature.")
    if len(coordinates) != 2:
        raise ValueError("spatial_cols must contain x and y column names.")
    for name, columns in (
        ("feature_cols", features), ("spatial_cols", coordinates),
        ("index_cols", keys), ("obs_cols", metadata),
    ):
        if any(not isinstance(col, str) or not col for col in columns):
            raise TypeError(f"{name} must contain nonempty string column names.")
        if len(set(columns)) != len(columns):
            raise ValueError(f"{name} contains duplicate column names.")
    if set(features).intersection(coordinates):
        raise ValueError("Feature columns and coordinate columns must be distinct.")
    missing = set(features + coordinates + keys + metadata).difference(df.columns)
    if missing:
        raise KeyError(f"Columns missing from input table: {sorted(missing)}")
    for col in features + coordinates:
        dtype = df[col].dtype
        if not is_numeric_dtype(dtype) or is_complex_dtype(dtype) or is_bool_dtype(dtype):
            raise TypeError(f"Expected a real numeric column for {col!r}, got {dtype}.")

    if keys:
        identifiers = df[keys]
        if identifiers.isna().to_numpy().any():
            raise ValueError("Cell identifiers contain missing values.")
        if len(keys) == 1:
            names = identifiers.iloc[:, 0].astype(str)
        else:
            names = [
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                for row in identifiers.astype(str).itertuples(index=False, name=None)
            ]
    else:
        source_index = df.index
        has_missing_id = (
            any((codes == -1).any() for codes in source_index.codes)
            if isinstance(source_index, pd.MultiIndex)
            else source_index.hasnans
        )
        if has_missing_id:
            raise ValueError("The input index contains missing cell identifiers.")
        names = source_index.map(str)
    index = pd.Index(names, dtype=str)
    if not index.is_unique:
        raise ValueError("Cell identifiers must be unique. Specify unique index_cols.")

    obs = df[metadata].copy()
    obs.index = index
    return AnnData(
        X=df[features].to_numpy(dtype=np.float64, na_value=np.nan, copy=True),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(features, dtype=str)),
        obsm={
            "spatial": df[coordinates].to_numpy(
                dtype=np.float64, na_value=np.nan, copy=True
            ),
        },
    )
