"""Render a linked cell segmentation with Matplotlib."""

# Portions of this file were refined with LLM assistance.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, LogNorm, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.transforms import nonsingular
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .mask import _image_rows, _read_label_mask, _validate_label_mask

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes


def _rgba8(color) -> np.ndarray:
    """Convert a Matplotlib color specification to four 8-bit channels."""
    return np.rint(255 * np.asarray(to_rgba(color))).astype(np.uint8)


def _contrast_text_color(
    background: tuple[float, float, float, float],
    backdrop: tuple[float, float, float, float] | None = None,
) -> str | None:
    """Choose the higher-contrast text color for a known sRGB background."""
    rgb = np.asarray(background[:3])
    alpha = background[3]
    if alpha < 1:
        if backdrop is None or backdrop[3] < 1:
            return None
        rgb = alpha * rgb + (1 - alpha) * np.asarray(backdrop[:3])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = linear @ (0.2126, 0.7152, 0.0722)
    # Equal contrast ratios for black and white text.
    switch = (1.05 * 0.05) ** 0.5 - 0.05
    return "white" if luminance < switch else "black"


def _categorical_colors(values, category_source, cmap, palette):
    """Map nonmissing categories to colors and build their legend entries."""
    categories = pd.Categorical(category_source).categories
    codes = pd.Categorical(values, categories=categories).codes
    if cmap is None:
        swatches = np.concatenate([
            plt.get_cmap(name)(np.arange(20), bytes=True)
            for name in ("tab20", "tab20b", "tab20c")
        ])
        colors = swatches[np.arange(len(categories)) % len(swatches)]
    else:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(categories)), bytes=True)
    if palette is not None:
        for i, category in enumerate(categories):
            if category in palette:
                colors[i] = _rgba8(palette[category])

    valid = codes >= 0
    handles = [
        Patch(facecolor=colors[i] / 255, edgecolor="none", label=str(categories[i]))
        for i in np.unique(codes[valid])
    ]
    return valid, colors[codes[valid]], handles


def _continuous_colors(values, cmap, norm, unmatched_color):
    """Color finite scalar values and retain the mappable for the colorbar."""
    numeric = values.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(numeric)
    if isinstance(norm, LogNorm):
        valid &= numeric > 0
    if not valid.any():
        return valid, np.empty((0, 4), dtype=np.uint8), None

    color_map = plt.get_cmap("viridis" if cmap is None else cmap)
    mapper = ScalarMappable(cmap=color_map.with_extremes(bad=unmatched_color), norm=norm)
    mapper.set_array(numeric[valid])
    mapper.autoscale_None()
    if mapper.norm.vmin == mapper.norm.vmax:
        # Keep constant-value images and their colorbars on the same scale.
        low, high = nonsingular(mapper.norm.vmin, mapper.norm.vmax, expander=0.1)
        mapper.set_clim(low, high)
    return valid, mapper.to_rgba(numeric[valid], bytes=True), mapper


def _add_outline_collection(ax, segments, style):
    """Draw one outline collection without expanding the image extent."""
    if len(segments):
        options = {
            "colors": "white", "linewidths": 0.35, "capstyle": "projecting",
            "rasterized": True, "zorder": 2,
        }
        options.update(style)
        ax.add_collection(LineCollection(segments, **options), autolim=False)


def _add_outline(ax, mask, style):
    """Draw each shared pixel edge once, including the image perimeter."""
    padded = np.pad(mask, 1)
    row, col = np.nonzero(padded[:-1, 1:-1] != padded[1:, 1:-1])
    starts = np.column_stack((col, row))
    horizontal = np.stack((starts, starts + (1, 0)), axis=1)
    row, col = np.nonzero(padded[1:-1, :-1] != padded[1:-1, 1:])
    starts = np.column_stack((col, row))
    vertical = np.stack((starts, starts + (0, 1)), axis=1)
    _add_outline_collection(ax, np.concatenate((horizontal, vertical)), style)


def _cell_crops(mask, labels, pixel_codes):
    """Yield each cell's binary bounding box and its row and column offsets."""
    rows, cols = np.nonzero(mask)
    codes = pixel_codes[rows, cols]
    # Compact codes allow sparse label IDs without large bounding-box arrays.
    top = np.full(labels.size, mask.shape[0], dtype=np.intp)
    left = np.full(labels.size, mask.shape[1], dtype=np.intp)
    bottom = np.zeros(labels.size, dtype=np.intp)
    right = np.zeros(labels.size, dtype=np.intp)
    np.minimum.at(top, codes, rows)
    np.minimum.at(left, codes, cols)
    np.maximum.at(bottom, codes, rows + 1)
    np.maximum.at(right, codes, cols + 1)
    for code in np.flatnonzero(labels):
        y, x = top[code], left[code]
        yield mask[y:bottom[code], x:right[code]] == labels[code], y, x


def _add_contour_outline(ax, regions, style):
    """Extract interpolated paths and draw them as one collection."""
    # A private axes keeps contour extraction from changing the plotted limits.
    work_ax = Figure().subplots()
    paths = []
    for region, y, x in regions:
        if not region.any():
            continue
        # Padding closes contours at the crop boundary. Pixel centers are
        # half-integer coordinates in the mask display frame.
        padded = np.pad(region, 1)
        x_centers = x + np.arange(padded.shape[1]) - 0.5
        y_centers = y + np.arange(padded.shape[0]) - 0.5
        contour = work_ax.contour(x_centers, y_centers, padded, levels=[0.5])
        paths.extend(path for path in contour.allsegs[0] if len(path) > 1)
        if hasattr(contour, "remove"):
            contour.remove()
        else:
            # Older Matplotlib releases expose the artists as collections.
            for artist in contour.collections:
                artist.remove()
    _add_outline_collection(ax, paths, {"joinstyle": "round", **style})


def plot_mask(
    adata: AnnData,
    *,
    image_id: str | int,
    color: str,
    image_col: str = "ImageNumber",
    mask_id_col: str = "mask_id",
    mask: str | Path | np.ndarray | None = None,
    layer: str | None = None,
    cmap: str | Colormap | None = None,
    palette: Mapping[Any, Any] | None = None,
    categorical: bool | None = None,
    norm: Normalize | None = None,
    background_color: Any = "white",
    unmatched_color: Any = "#D0D0D0",
    outline: bool = True,
    outline_mode: Literal["pixel", "contour"] = "pixel",
    outline_kws: Mapping[str, Any] | None = None,
    outer_outline_kws: Mapping[str, Any] | None = None,
    show_legend: bool = True,
    legend_kws: Mapping[str, Any] | None = None,
    frame_kws: Mapping[str, Any] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6, 6),
    title: str | None = None,
    title_color: Any = None,
) -> tuple[Figure, Axes]:
    """Render one segmentation image using its assigned cell labels.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData containing image IDs and assigned mask labels in ``obs``.
    image_id : str or int
        Value in ``obs[image_col]`` identifying the image to display.
    color : str
        Observation column or feature name. Observation columns take
        precedence unless a feature layer is specified.
    image_col : str, default='ImageNumber'
        Observation column identifying each image.
    mask_id_col : str, default='mask_id'
        Observation column containing the original mask labels.
    mask : str, pathlib.Path, numpy.ndarray, or None, default=None
        TIFF path or 2D integer label array. The default uses the path in
        ``uns["mask_paths"]`` registered by ``assign_mask_ids``.
    layer : str or None, default=None
        Layer from which to read feature values. The default reads ``X``
        when coloring by a feature.
    cmap : str, matplotlib.colors.Colormap, or None, default=None
        Matplotlib colormap. The defaults are the tab palettes for
        categorical data and viridis for continuous data.
    palette : mapping or None, default=None
        Explicit colors keyed by category value. Categories without an
        override retain their colors from the colormap.
    categorical : bool or None, default=None
        Whether to interpret values as categories. The default infers
        categories for nonnumeric or Boolean values and when a palette
        is supplied. Set True for integer cluster identifiers.
    norm : matplotlib.colors.Normalize or None, default=None
        Normalization for continuous colors. The default uses values
        visible in this image. Pass fixed limits to compare images.
    background_color : color, default='white'
        Color for label 0, the legend background, and a newly created
        figure canvas.
    unmatched_color : color, default='#D0D0D0'
        Color for segments without a linked cell or usable annotation.
    outline : bool, default=True
        Draw the boundaries between individual labeled regions.
    outline_mode : {'pixel', 'contour'}, default='pixel'
        Draw cell and foreground outlines along pixel edges or as
        interpolated 0.5-level contours. Cell contours are extracted from
        each label's bounding box.
    outline_kws : mapping or None, default=None
        Matplotlib LineCollection options for cell boundaries.
    outer_outline_kws : mapping or None, default=None
        If supplied, draw a foreground outline using these LineCollection
        options.
    show_legend : bool, default=True
        Add a categorical legend or a continuous colorbar when applicable.
    legend_kws : mapping or None, default=None
        Options for the categorical legend or continuous colorbar. Use
        ``labelcolor`` to override the automatic text color.
    frame_kws : mapping or None, default=None
        Figure patch properties, such as ``edgecolor`` and ``linewidth``.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on. If supplied, the canvas color is controlled by
        the enclosing figure and any explicit ``frame_kws``.
    figsize : tuple of float, default=(6, 6)
        Size of a newly created figure in inches.
    title : str or None, default=None
        Optional title for the mask panel.
    title_color : color or None, default=None
        Explicit title color. The default uses the Matplotlib title color
        if configured, or selects black or white based on the canvas.

    Returns
    -------
    tuple of matplotlib.figure.Figure and matplotlib.axes.Axes
        Figure and axes containing the rendered mask.

    Notes
    -----
    The display extent follows the zero-based mask pixel frame. For
    transparent backgrounds with no known backdrop, text retains the
    Matplotlib style.
    """
    if outline_mode not in ("pixel", "contour"):
        raise ValueError("outline_mode must be 'pixel' or 'contour'.")
    if not adata.obs.columns.is_unique:
        raise ValueError("adata.obs must have unique column names.")
    rows, image_key = _image_rows(adata.obs[image_col], image_id)
    if mask is None:
        try:
            mask = adata.uns["mask_paths"][image_key]
        except KeyError as exc:
            raise KeyError(f"No TIFF path for image {image_key!r}. Pass mask= explicitly.") from exc
    if isinstance(mask, (str, Path)):
        mask = _read_label_mask(Path(mask).expanduser())
    else:
        mask = _validate_label_mask(np.asarray(mask))

    # Compact label codes avoid allocating a lookup table up to the largest ID.
    labels, pixel_codes = np.unique(mask, return_inverse=True)
    pixel_codes = pixel_codes.reshape(mask.shape)
    mask_ids = pd.array(adata.obs[mask_id_col].iloc[rows], dtype="Int64")
    mask_ids = mask_ids.to_numpy(dtype=np.int64, na_value=0)
    positions = pd.Index(labels.astype(np.int64, copy=False)).get_indexer(mask_ids)
    linked = (mask_ids > 0) & (positions >= 0)
    if not pd.Index(mask_ids[linked]).is_unique:
        raise ValueError(f"Multiple cells share a mask ID in image {image_key!r}.")
    cell_rows, positions = rows[linked], positions[linked]

    if layer is None and color in adata.obs:
        category_source = adata.obs[color]
        values = category_source.iloc[cell_rows]
    else:
        if color not in adata.var_names:
            raise KeyError(f"{color!r} is not available as a feature in adata.var_names.")
        cells = adata[cell_rows, [color]]
        matrix = cells.X if layer is None else cells.layers[layer]
        if matrix is None:
            raise ValueError("adata.X does not contain a feature matrix.")
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        category_source = values = pd.Series(np.asarray(matrix).ravel(), name=color)

    if categorical is None:
        categorical = (
            palette is not None or is_bool_dtype(values.dtype)
            or not is_numeric_dtype(values.dtype)
        )
    if categorical and norm is not None:
        raise ValueError("norm applies only to continuous values.")
    if not categorical and palette is not None:
        raise ValueError("palette applies only to categorical values.")

    lookup = np.tile(_rgba8(unmatched_color), (len(labels), 1))
    lookup[labels == 0] = _rgba8(background_color)
    if categorical:
        valid, colors, handles = _categorical_colors(values, category_source, cmap, palette)
    else:
        valid, colors, mapper = _continuous_colors(values, cmap, norm, unmatched_color)
    lookup[positions[valid]] = colors

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=background_color)
    else:
        fig = ax.figure
    if frame_kws is not None:
        fig.patch.set(**frame_kws)
    canvas = fig.get_facecolor() if fig.patch.get_visible() else to_rgba("none")
    canvas_text_color = _contrast_text_color(canvas)
    height, width = mask.shape
    ax.imshow(
        lookup[pixel_codes],
        extent=(0, width, height, 0),
        origin="upper", interpolation="nearest", aspect="equal",
    )
    if outline:
        style = dict(outline_kws or {})
        if outline_mode == "contour":
            _add_contour_outline(ax, _cell_crops(mask, labels, pixel_codes), style)
        else:
            _add_outline(ax, mask, style)
    if outer_outline_kws is not None:
        foreground = mask > 0
        style = {"zorder": 3, **outer_outline_kws}
        if outline_mode == "contour":
            _add_contour_outline(ax, [(foreground, 0, 0)], style)
        else:
            _add_outline(ax, foreground, style)
    ax.set_axis_off()
    if title is not None:
        title_artist = ax.set_title(title)
        if title_color is not None:
            title_artist.set_color(title_color)
        elif canvas_text_color is not None and plt.rcParams["axes.titlecolor"] == "auto":
            title_artist.set_color(canvas_text_color)

    if show_legend:
        if categorical:
            if valid.sum() < np.count_nonzero(labels):
                handles.append(Patch(facecolor=unmatched_color, edgecolor="none", label="Uncolored"))
            if handles:
                options = {
                    "loc": "center left", "bbox_to_anchor": (1.02, 0.5),
                    "frameon": True, "facecolor": background_color, "edgecolor": "none",
                    **(legend_kws or {}),
                }
                legend = ax.legend(handles=handles, **options)
                legend_frame = legend.get_frame()
                if "framealpha" not in options:
                    # Honor any alpha in the face and edge colors.
                    legend_frame.set_alpha(None)
                legend_face = legend_frame.get_facecolor() if legend.get_frame_on() else canvas
                legend_text_color = _contrast_text_color(legend_face, canvas)
                if legend_text_color is not None:
                    style_color = plt.rcParams["legend.labelcolor"]
                    if "labelcolor" not in options and style_color in (None, "None"):
                        for text in legend.get_texts():
                            text.set_color(legend_text_color)
                    legend.get_title().set_color(legend_text_color)
        elif mapper is not None:
            options = {"label": color, **(legend_kws or {})}
            labelcolor = options.pop("labelcolor", canvas_text_color)
            colorbar = fig.colorbar(mapper, ax=ax, **options)
            colorbar.ax.set_facecolor(background_color)
            if labelcolor is not None:
                vertical = colorbar.orientation == "vertical"
                axis = colorbar.ax.yaxis if vertical else colorbar.ax.xaxis
                axis.label.set_color(labelcolor)
                axis.get_offset_text().set_color(labelcolor)
                colorbar.ax.tick_params(axis="y" if vertical else "x", labelcolor=labelcolor)
    return fig, ax
