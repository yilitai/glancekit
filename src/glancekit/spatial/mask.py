"""Link single-cell measurements to a 2D instance segmentation."""

# Portions of this file were refined with LLM assistance.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_float_dtype,
    is_integer_dtype,
    is_object_dtype,
    is_unsigned_integer_dtype,
)

if TYPE_CHECKING:
    from anndata import AnnData


def _image_rows(values: pd.Series, image_id: str | int) -> tuple[np.ndarray, str]:
    """Return the selected row positions and an unambiguous H5AD-safe key."""
    rows = np.flatnonzero(values.eq(image_id).fillna(False).to_numpy(dtype=bool))
    if not rows.size:
        raise ValueError(f"No cells with {values.name}={image_id!r}.")
    stored_ids = values.iloc[rows].astype(str)
    key = stored_ids.iloc[0]
    if not stored_ids.eq(key).all():
        raise ValueError("The selected image has inconsistent string IDs.")
    if not key or "/" in key or "\0" in key or key in {".", ".."}:
        raise ValueError(f"Image ID {key!r} is not a valid H5AD mapping key.")
    if isinstance(values.dtype, pd.CategoricalDtype):
        other_ids = values.cat.categories
    elif is_object_dtype(values.dtype):
        other_ids = values.dropna().unique()
    else:
        other_ids = ()
    if any(value != image_id and str(value) == key for value in other_ids):
        raise ValueError(f"Different image IDs serialize to {key!r}.")
    return rows, key


def _mask_image_column(adata: AnnData, image_col: str | None) -> str:
    """Resolve the image column consistently across the mask workflow."""
    registered = adata.uns.get("mask_image_col")
    if registered is not None and image_col is not None and image_col != registered:
        raise ValueError(f"Mask paths are registered using image_col={registered!r}.")
    return image_col if image_col is not None else (
        registered if registered is not None else "ImageNumber"
    )


def _select_mask_rows(
    adata: AnnData, image_col: str | None, image_id: str | int | None
) -> tuple[np.ndarray, str | None]:
    """Select one section; never silently combine known image groups."""
    column = _mask_image_column(adata, image_col)
    paths = adata.uns.get("mask_paths", {})
    if image_id is not None:
        return _image_rows(adata.obs[column], image_id)
    if len(paths) > 1:
        raise ValueError("Select one section with image_id.")
    rows = np.arange(adata.n_obs, dtype=np.intp)
    if column in adata.obs:
        values = adata.obs[column]
        if values.nunique(dropna=False) > 1:
            raise ValueError("Select one section with image_id.")
        if paths and len(values):
            if values.isna().any():
                raise ValueError("Missing image ID; select a section explicitly.")
            selected, key = _image_rows(values, values.iloc[0])
            if key not in paths:
                raise ValueError("The selected section does not match the registered mask path. "
                                 "Specify image_id explicitly.")
            return selected, key
    elif image_col is not None or adata.uns.get("mask_image_col") is not None:
        raise KeyError(f"Image column {column!r} is missing from adata.obs.")
    return rows, None


def _resolve_section_mask(adata: AnnData, rows: np.ndarray, image_key: str | None, mask):
    """Resolve an explicit mask or the registered path for the selected section."""
    if mask is not None:
        return mask
    paths = adata.uns.get("mask_paths", {})
    if image_key is not None:
        mask = paths.get(image_key)
        if mask is None and not paths and len(rows) == adata.n_obs:
            mask = adata.uns.get("mask_path")
    elif len(paths) == 1:
        mask = next(iter(paths.values()))
    else:
        mask = adata.uns.get("mask_path")
    if mask is None:
        raise KeyError("No mask path is registered for the section. Pass mask explicitly.")
    return mask


def _invalidated_morphology(adata: AnnData, rows: np.ndarray):
    """Prepare invalidation of default and registered derived morphology tables."""
    keys = {"morphology", *adata.uns.get("_morphy_morphology_keys", [])}
    updates = {}
    for key in keys:
        table = adata.obsm.get(key)
        if table is None:
            continue
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"adata.obsm[{key!r}] must contain a morphology DataFrame.")
        updated = table.copy()
        updated.iloc[rows, :] = np.nan
        updates[key] = updated
    return updates


def _read_label_mask(path: Path) -> np.ndarray:
    """Read a single 2D TIFF series as an instance label array."""
    from tifffile import TiffFile

    with TiffFile(path) as tif:
        if len(tif.series) != 1 or len(tif.series[0].shape) != 2:
            raise ValueError("Expected one 2D instance label image in the TIFF.")
        mask = tif.series[0].asarray()
    return _validate_label_mask(mask)


def _validate_label_mask(mask: np.ndarray) -> np.ndarray:
    """Require a nonempty, nonnegative integer mask representable as int64."""
    if mask.ndim != 2 or mask.dtype.kind not in "iu" or not mask.size:
        raise ValueError("Expected a nonempty 2D array of integer instance labels.")
    if mask.min() < 0 or int(mask.max()) > np.iinfo(np.int64).max:
        raise ValueError("Mask labels must be nonnegative and fit in int64.")
    return mask


def _labels_at_coordinates(
    spatial: np.ndarray | pd.DataFrame, rows: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample labels at cell coordinates and count background or invalid hits."""
    if isinstance(spatial, pd.DataFrame):
        coordinates = spatial.iloc[rows].to_numpy(na_value=np.nan, dtype=float)
    else:
        coordinates = np.asarray(spatial[rows])
    if coordinates.shape != (rows.size, 2) or coordinates.dtype.kind not in "iuf":
        raise ValueError("Spatial coordinates must be a numeric (n_cells, 2) array.")

    x, y = coordinates.T
    height, width = mask.shape
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    labels = np.zeros(rows.size, dtype=np.int64)
    # Valid coordinates are nonnegative, so integer conversion is flooring.
    labels[valid] = mask[y[valid].astype(np.intp), x[valid].astype(np.intp)]
    return labels, {
        "invalid_coordinates": int((~valid).sum()),
        "background": int((valid & (labels == 0)).sum()),
    }


def _labels_from_column(
    source: pd.Series, mask: np.ndarray
) -> tuple[np.ndarray, dict[str, int]]:
    """Verify supplied label IDs against the positive labels in the mask."""
    dtype = source.dtype
    if not (is_integer_dtype(dtype) or is_float_dtype(dtype)):
        raise TypeError(f"{source.name!r} must contain numeric integer label IDs.")
    if is_unsigned_integer_dtype(dtype) and (source > np.iinfo(np.int64).max).any():
        raise ValueError(f"{source.name!r} contains IDs outside the int64 range.")
    if is_float_dtype(dtype) and (source.abs() >= 2**53).any():
        raise ValueError(f"Use an integer dtype for large label IDs in {source.name!r}.")
    try:
        labels = pd.array(source, dtype="Int64").to_numpy(dtype=np.int64, na_value=0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{source.name!r} must contain integer-valued IDs.") from exc
    positive = labels > 0
    present = positive & np.isin(labels, np.unique(mask).astype(np.int64, copy=False))
    return np.where(present, labels, 0), {
        "invalid_labels": int((~positive).sum()),
        "absent_labels": int((positive & ~present).sum()),
    }


def assign_mask_ids(
    adata: AnnData,
    mask_path: str | Path,
    *,
    image_col: str | None = None,
    image_id: str | int | None = None,
    spatial_key: str = "spatial",
    key_added: str = "mask_id",
    label_col: str | None = None,
) -> dict[str, int]:
    """Link cells from one image to segmentation labels in place.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData containing the target image's cells. Pass a full AnnData
        object or an explicit copy of a view.
    mask_path : str or pathlib.Path
        TIFF containing one nonempty, two-dimensional integer instance mask.
        Label 0 denotes background.
    image_col : str or None, default=None
        Column in ``obs`` that identifies each image. Defaults to the
        registered image column, or ``"ImageNumber"`` if none is registered.
    image_id : str, int or None, default=None
        Value in ``obs[image_col]`` identifying the image to update. Required
        when multiple sections are known; otherwise it may be omitted.
    spatial_key : str, default='spatial'
        Key in ``obsm`` containing zero-based (x, y) pixel coordinates in
        the TIFF frame. Used when ``label_col`` is absent.
    key_added : str, default='mask_id'
        Output column in ``obs``. It stores the original mask labels as
        nullable integers.
    label_col : str or None, default=None
        Column in ``obs`` known to contain numeric labels from this TIFF.
        The function checks that each positive label occurs in the mask.
        With the default, labels are sampled at ``mask[floor(y), floor(x)]``.

    Returns
    -------
    dict of str to int
        Counts for the selected image, including ``total``, ``matched``,
        ``unmatched``, and ``ambiguous``. The coordinate method also reports
        ``invalid_coordinates`` and ``background``. The label method reports
        ``invalid_labels`` and ``absent_labels``.

    Notes
    -----
    Repeated hits to the same nonzero label are ambiguous. Unmatched cells
    receive missing values, including when no cell in the image matches.
    Assignments for other images are retained when selecting an image.
    The TIFF's absolute path is stored in ``uns["mask_path"]`` for a single
    section, or in ``uns["mask_paths"]`` under the string image ID when
    selecting an image. Use the same image and output columns across cohort calls.
    Previously computed morphology for the selected rows is invalidated,
    including custom output tables registered by linked_cell_morphology.
    """
    if adata.is_view:
        raise ValueError("Pass an AnnData object or an explicit copy of a view.")
    if not adata.obs.columns.is_unique:
        raise ValueError("adata.obs must have unique column names.")
    if key_added in (_mask_image_column(adata, image_col), label_col):
        raise ValueError("key_added must differ from the image and source label columns.")
    rows, image_key = _select_mask_rows(adata, image_col, image_id)
    paths = dict(adata.uns.get("mask_paths", {}))
    path = Path(mask_path).expanduser().resolve(strict=True)
    mask = _read_label_mask(path)
    if label_col is None:
        labels, counts = _labels_at_coordinates(adata.obsm[spatial_key], rows, mask)
    else:
        labels, counts = _labels_from_column(adata.obs[label_col].iloc[rows], mask)

    positive = labels > 0
    ambiguous = positive & pd.Index(labels).duplicated(keep=False)
    matched = positive & ~ambiguous
    n_matched = int(matched.sum())
    assigned = pd.array(labels, dtype="Int64")
    assigned[~matched] = pd.NA
    invalidated = _invalidated_morphology(adata, rows)
    existing = adata.obs.get(key_added)
    if existing is not None and existing.dtype == pd.Int64Dtype():
        adata.obs.iloc[rows, adata.obs.columns.get_loc(key_added)] = assigned
    else:
        output = pd.array(
            np.zeros(adata.n_obs, dtype=np.int64) if existing is None else existing,
            dtype="Int64",
            copy=True,
        )
        if existing is None:
            output[:] = pd.NA
        output[rows] = assigned
        adata.obs[key_added] = output
    if image_key is None:
        adata.uns["mask_path"] = str(path)
    else:
        paths[image_key] = str(path)
        adata.uns["mask_paths"] = paths
        adata.uns["mask_image_col"] = _mask_image_column(adata, image_col)
        # The named registry supersedes the old unqualified path.
        adata.uns.pop("mask_path", None)
    for morphology_key, table in invalidated.items():
        adata.obsm[morphology_key] = table
    return {
        "total": int(rows.size),
        "matched": n_matched,
        "unmatched": int(rows.size) - n_matched,
        "ambiguous": int(ambiguous.sum()),
        **counts,
    }
