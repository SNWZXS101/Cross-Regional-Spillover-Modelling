# -*- coding: utf-8 -*-
"""
plot_GDGX_spillover_map_v3b.py

High-contrast watercolor + light real-terrain version for the
Guangdong–Guangxi spillover study.

Key improvements over v3
------------------------
1. Larger, clearer serif labels with stronger white halo.
2. Dark gray/near-black Guangdong–Guangxi outlines.
3. Higher-contrast neighboring provincial boundaries.
4. Stronger localized watercolor / ink-wash texture.
5. Optional REAL shaded-relief background from Natural Earth SR_LR
   (derived by Natural Earth from downsampled SRTM Plus elevation data).
6. Optional Natural Earth river centerlines for a subtle hydrographic layer.
7. Compact, non-overlapping spillover arrows.
8. Journal-source note written automatically.

Required packages
-----------------
numpy
matplotlib
geopandas
shapely
pyproj
Pillow (normally installed with matplotlib)

No scipy, rasterio, requests, or cartopy are required.

Run
---
python plot_GDGX_spillover_map_v3b.py

Outputs
-------
south_china_spillover_map_v3.png
south_china_spillover_map_v3.pdf
south_china_spillover_map_v3.svg
map_source_note_v3.txt
"""

from __future__ import annotations

import io
import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from PIL import Image
from matplotlib.path import Path as MplPath
from matplotlib.patches import (
    PathPatch,
    FancyArrowPatch,
    Rectangle,
    Polygon as MplPolygon,
)
from matplotlib.colors import to_rgb
from shapely.geometry import Polygon, MultiPolygon, box
from pyproj import Geod


# ============================================================
# 1. USER SETTINGS
# ============================================================

DATA_DIR = Path("map_data")
DATA_DIR.mkdir(exist_ok=True)

OUT_PNG = "south_china_spillover_map_v3b.png"
OUT_PDF = "south_china_spillover_map_v3b.pdf"
OUT_SVG = "south_china_spillover_map_v3b.svg"
OUT_NOTE = "map_source_note_v3b.txt"

# Toggle the real relief / river layers here.
USE_REAL_RELIEF = True
USE_RIVERS = True

# Relief transparency. 0.12–0.24 is usually appropriate.
RELIEF_ALPHA = 0.20

# River transparency and linewidth.
RIVER_ALPHA = 0.45
RIVER_WIDTH = 0.55

# Main output size.
FIGSIZE = (11.1, 4.65)

# ------------------------------------------------------------
# Natural Earth downloads
# ------------------------------------------------------------

ADMIN1_URL = (
    "https://naturalearth.s3.amazonaws.com/"
    "10m_cultural/ne_10m_admin_1_states_provinces.zip"
)
ADMIN0_URL = (
    "https://naturalearth.s3.amazonaws.com/"
    "10m_cultural/ne_10m_admin_0_countries.zip"
)
RIVERS_URL = (
    "https://naturalearth.s3.amazonaws.com/"
    "10m_physical/ne_10m_rivers_lake_centerlines.zip"
)
RELIEF_URL = (
    "https://naturalearth.s3.amazonaws.com/"
    "10m_raster/SR_LR.zip"
)

ADMIN1_ZIP = DATA_DIR / "ne_10m_admin_1_states_provinces.zip"
ADMIN0_ZIP = DATA_DIR / "ne_10m_admin_0_countries.zip"
RIVERS_ZIP = DATA_DIR / "ne_10m_rivers_lake_centerlines.zip"
RELIEF_ZIP = DATA_DIR / "SR_LR.zip"
RELIEF_DIR = DATA_DIR / "SR_LR"

# ============================================================
# 2. PALETTE
# ============================================================

TEXT = "#132A69"

# Background land / boundaries
LAND = "#F2F2F2"
BACKGROUND_BOUNDARY = "#A8A8A8"
MAIN_BOUNDARY = "#3F3F3F"

# Guangdong — brighter clear blue
GD_LIGHT = "#E3EDFF"
GD_MID   = "#AFC8F6"
GD_DEEP  = "#7196E7"

# Guangxi — sage / jade green
GX_LIGHT = "#E3F0E7"
GX_MID   = "#B5D9C0"
GX_DEEP  = "#72AC8D"

# Arrow colors
ARROW_BLUE = "#5878D2"
ARROW_GREEN = "#5F9E84"

# Rivers
RIVER_COLOR = "#6A9FC7"

# Text sizes
REGION_FS = 18.5
SPILLOVER_FS = 14.6
INSET_FS = 12.0
SMALL_FS = 10.4

FONT_FAMILY = "Times New Roman"


# ============================================================
# 3. DOWNLOAD HELPERS
# ============================================================

def download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return

    print(f"Downloading:\n  {url}\n-> {path}")
    urllib.request.urlretrieve(url, str(path))


def ensure_relief_extracted() -> tuple[Path, Path] | tuple[None, None]:
    """
    Download and extract Natural Earth shaded relief (SR_LR).
    Returns paths to tif and tfw.
    """
    if not USE_REAL_RELIEF:
        return None, None

    download_if_missing(RELIEF_URL, RELIEF_ZIP)
    RELIEF_DIR.mkdir(exist_ok=True)

    tif_candidates = list(RELIEF_DIR.rglob("*.tif"))
    tfw_candidates = list(RELIEF_DIR.rglob("*.tfw"))

    if not tif_candidates or not tfw_candidates:
        print("Extracting Natural Earth relief...")
        with zipfile.ZipFile(RELIEF_ZIP, "r") as zf:
            zf.extractall(RELIEF_DIR)

        tif_candidates = list(RELIEF_DIR.rglob("*.tif"))
        tfw_candidates = list(RELIEF_DIR.rglob("*.tfw"))

    if not tif_candidates or not tfw_candidates:
        print("WARNING: relief TIFF/TFW not found; continuing without relief.")
        return None, None

    return tif_candidates[0], tfw_candidates[0]


# ============================================================
# 4. NATURAL EARTH SCHEMA HELPERS
# ============================================================

def get_china_admin0(admin0: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tests = [
        ("ADM0_A3", "CHN"),
        ("SOV_A3", "CH1"),
        ("ADMIN", "China"),
        ("NAME_EN", "China"),
        ("NAME", "China"),
    ]

    for col, val in tests:
        if col not in admin0.columns:
            continue

        if col in {"ADM0_A3", "SOV_A3"}:
            out = admin0[admin0[col].astype(str).eq(val)].copy()
        else:
            out = admin0[
                admin0[col].astype(str).str.contains(
                    val, case=False, na=False, regex=False
                )
            ].copy()

        if not out.empty:
            return out

    raise RuntimeError("Could not identify China in Admin-0 data.")


def get_china_admin1(admin1: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for col in ["adm0_a3", "ADM0_A3", "sr_adm0_a3"]:
        if col in admin1.columns:
            out = admin1[admin1[col].astype(str).eq("CHN")].copy()
            if not out.empty:
                return out

    for col in ["admin", "geonunit"]:
        if col in admin1.columns:
            out = admin1[
                admin1[col].astype(str).str.contains(
                    "China", case=False, na=False, regex=False
                )
            ].copy()
            if not out.empty:
                return out

    raise RuntimeError("Could not identify China Admin-1 provinces.")


def find_province(df: gpd.GeoDataFrame, keyword: str) -> gpd.GeoDataFrame:
    cols = [
        "name",
        "name_en",
        "gn_name",
        "woe_name",
        "name_local",
    ]

    for col in cols:
        if col not in df.columns:
            continue

        m = df[col].astype(str).str.contains(
            keyword, case=False, na=False, regex=False
        )

        if m.any():
            return df.loc[m].copy()

    raise RuntimeError(f"Province '{keyword}' not found.")


def geom_union(gdf: gpd.GeoDataFrame):
    try:
        return gdf.geometry.union_all()
    except AttributeError:
        return gdf.geometry.unary_union


# ============================================================
# 5. SHAPELY -> MATPLOTLIB PATCH
# ============================================================

def polygon_to_path(poly: Polygon) -> MplPath:
    vertices = []
    codes = []

    ext = np.asarray(poly.exterior.coords)
    vertices.extend(ext.tolist())
    codes.extend(
        [MplPath.MOVETO]
        + [MplPath.LINETO] * (len(ext) - 2)
        + [MplPath.CLOSEPOLY]
    )

    for interior in poly.interiors:
        arr = np.asarray(interior.coords)
        vertices.extend(arr.tolist())
        codes.extend(
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(arr) - 2)
            + [MplPath.CLOSEPOLY]
        )

    return MplPath(np.asarray(vertices), np.asarray(codes))


def geometry_patch(geom, **kwargs) -> PathPatch:
    if isinstance(geom, Polygon):
        return PathPatch(polygon_to_path(geom), **kwargs)

    if isinstance(geom, MultiPolygon):
        paths = [
            polygon_to_path(g)
            for g in geom.geoms
            if not g.is_empty
        ]
        compound = MplPath.make_compound_path(*paths)
        return PathPatch(compound, **kwargs)

    raise TypeError(f"Unsupported geometry: {geom.geom_type}")


# ============================================================
# 6. REAL NATURAL EARTH RELIEF
# ============================================================

def read_tfw(tfw_path: Path) -> tuple[float, float, float, float, float, float]:
    vals = []
    with open(tfw_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(float(line))

    if len(vals) < 6:
        raise RuntimeError(f"Invalid world file: {tfw_path}")

    # A, D, B, E, C, F
    return tuple(vals[:6])


def crop_relief(
    tif_path: Path,
    tfw_path: Path,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
):
    """
    Crop global Natural Earth SR_LR raster to the desired lon/lat bounds.

    SR_LR is north-up with a standard world file, so no rasterio is needed.
    """
    A, D, B, E, C, F = read_tfw(tfw_path)

    if abs(D) > 1e-12 or abs(B) > 1e-12:
        raise RuntimeError("Rotated raster world file not supported.")

    img = Image.open(tif_path).convert("L")
    W, H = img.size

    def lon_to_col(lon):
        return (lon - C) / A

    def lat_to_row(lat):
        return (lat - F) / E

    c0 = int(np.floor(min(lon_to_col(lon_min), lon_to_col(lon_max)))) - 2
    c1 = int(np.ceil(max(lon_to_col(lon_min), lon_to_col(lon_max)))) + 2

    r0 = int(np.floor(min(lat_to_row(lat_min), lat_to_row(lat_max)))) - 2
    r1 = int(np.ceil(max(lat_to_row(lat_min), lat_to_row(lat_max)))) + 2

    c0 = max(0, min(W - 1, c0))
    c1 = max(c0 + 1, min(W, c1))
    r0 = max(0, min(H - 1, r0))
    r1 = max(r0 + 1, min(H, r1))

    crop = img.crop((c0, r0, c1, r1))
    arr = np.asarray(crop, dtype=float) / 255.0

    # Pixel-center coordinates converted back to lon/lat.
    x_left = C + c0 * A
    x_right = C + (c1 - 1) * A
    y_top = F + r0 * E
    y_bottom = F + (r1 - 1) * E

    # Increase visual relief contrast mildly but keep it light.
    lo = np.percentile(arr, 3)
    hi = np.percentile(arr, 97)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-8), 0, 1)

    # Neutral light-gray terrain; invert slightly so darker = stronger relief.
    relief = 0.92 - 0.26 * (1.0 - arr)
    relief = np.clip(relief, 0, 1)

    return relief, [x_left, x_right, y_bottom, y_top]


# ============================================================
# 7. INK-WASH FILL
# ============================================================

def rgb(hexcolor: str) -> np.ndarray:
    return np.asarray(to_rgb(hexcolor), dtype=float)


def watercolor_fill(
    ax,
    geom,
    light_hex: str,
    mid_hex: str,
    deep_hex: str,
    seed: int,
    resolution: int = 650,
    alpha: float = 0.84,
    zorder: float = 5.0,
) -> None:
    """
    Multi-scale nonuniform color wash clipped to a province.

    This version intentionally creates stronger localized "ink pools"
    and more pale empty areas than v2.
    """
    rng = np.random.default_rng(seed)

    xmin, ymin, xmax, ymax = geom.bounds
    width = xmax - xmin
    height = ymax - ymin

    nx = resolution
    ny = max(220, int(resolution * height / max(width, 1e-12)))

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    field = np.zeros((ny, nx), dtype=float)

    # Large washes
    for _ in range(7):
        cx = rng.uniform(xmin + 0.06 * width, xmax - 0.06 * width)
        cy = rng.uniform(ymin + 0.06 * height, ymax - 0.06 * height)
        sx = rng.uniform(0.18, 0.38) * width
        sy = rng.uniform(0.18, 0.38) * height
        amp = rng.uniform(0.60, 1.10)

        field += amp * np.exp(
            -0.5 * (((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2)
        )

    # Local darker ink pools
    for _ in range(10):
        cx = rng.uniform(xmin, xmax)
        cy = rng.uniform(ymin, ymax)
        sx = rng.uniform(0.055, 0.13) * width
        sy = rng.uniform(0.055, 0.13) * height
        amp = rng.uniform(0.14, 0.34)

        field += amp * np.exp(
            -0.5 * (((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2)
        )

    field -= field.min()
    field /= field.max() + 1e-12
    field = field ** 1.38

    c0 = rgb(light_hex)
    c1 = rgb(mid_hex)
    c2 = rgb(deep_hex)

    rgba = np.ones((ny, nx, 4), dtype=float)

    split = 0.60
    low = field <= split
    high = ~low

    u = np.zeros_like(field)
    u[low] = field[low] / split

    v = np.zeros_like(field)
    v[high] = (field[high] - split) / (1.0 - split)

    for k in range(3):
        rgba[:, :, k][low] = c0[k] * (1 - u[low]) + c1[k] * u[low]
        rgba[:, :, k][high] = c1[k] * (1 - v[high]) + c2[k] * v[high]

    # Alpha variation gives "wash" behavior without hiding relief completely.
    rgba[:, :, 3] = alpha * (0.62 + 0.38 * field)

    clip = geometry_patch(
        geom,
        facecolor="none",
        edgecolor="none",
    )
    ax.add_patch(clip)

    im = ax.imshow(
        rgba,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        interpolation="bicubic",
        zorder=zorder,
    )
    im.set_clip_path(clip)


# ============================================================
# 8. RIVERS
# ============================================================

def add_rivers(
    ax,
    rivers: gpd.GeoDataFrame,
    bbox_geom,
    zorder=8,
):
    """
    Plot only rivers intersecting the map area.
    """
    if rivers is None or rivers.empty:
        return

    sub = rivers[rivers.intersects(bbox_geom)].copy()

    if sub.empty:
        return

    sub.plot(
        ax=ax,
        color=RIVER_COLOR,
        linewidth=RIVER_WIDTH,
        alpha=RIVER_ALPHA,
        zorder=zorder,
    )


# ============================================================
# 9. ARROWS / NORTH / SCALE
# ============================================================

def add_spillover_arrows(ax, cx, cy, w, h):
    x1 = cx - 0.107 * w
    x2 = cx + 0.107 * w

    y_upper = cy + 0.022 * h
    y_lower = cy - 0.015 * h

    # White underlay + colored arrow creates crisp separation.
    specs = [
        ((x1, y_upper), (x2, y_upper), ARROW_BLUE),
        ((x2, y_lower), (x1, y_lower), ARROW_GREEN),
    ]

    for p0, p1, col in specs:
        under = FancyArrowPatch(
            p0,
            p1,
            connectionstyle="arc3,rad=-0.42",
            arrowstyle="Simple,tail_width=10,head_width=23,head_length=23",
            facecolor="white",
            edgecolor="white",
            linewidth=0,
            alpha=0.98,
            zorder=11,
        )
        ax.add_patch(under)

        arr = FancyArrowPatch(
            p0,
            p1,
            connectionstyle="arc3,rad=-0.42",
            arrowstyle="Simple,tail_width=7.0,head_width=17.5,head_length=18.5",
            facecolor=col,
            edgecolor=col,
            linewidth=0,
            alpha=0.98,
            zorder=12,
        )
        ax.add_patch(arr)


def add_north_arrow(ax, x, y, w, h):
    nh = 0.115 * h
    hw = 0.014 * w

    poly = np.array([
        [x, y + nh],
        [x - hw, y],
        [x, y + 0.027 * h],
        [x + hw, y],
    ])

    ax.add_patch(
        MplPolygon(
            poly,
            closed=True,
            facecolor="white",
            edgecolor="#5B688F",
            linewidth=1.8,
            zorder=20,
        )
    )

    ax.text(
        x,
        y + nh + 0.016 * h,
        "N",
        ha="center",
        va="bottom",
        fontsize=SMALL_FS + 0.8,
        family=FONT_FAMILY,
        fontweight="bold",
        color=TEXT,
        zorder=21,
    )


def add_scale_bar(ax, x, y, center_lat, h):
    """
    0–250–500 km bar using geodesic distance in longitude direction.
    """
    geod = Geod(ellps="WGS84")

    # Determine longitude increment for 250 km at the selected latitude.
    lon0 = x
    lat0 = center_lat

    lon250, lat250, _ = geod.fwd(lon0, lat0, 90.0, 250_000.0)
    lon500, lat500, _ = geod.fwd(lon0, lat0, 90.0, 500_000.0)

    seg = lon250 - lon0
    bar_h = 0.010 * h

    ax.add_patch(
        Rectangle(
            (lon0, y),
            seg,
            bar_h,
            facecolor="#556184",
            edgecolor="#556184",
            linewidth=0.9,
            zorder=20,
        )
    )
    ax.add_patch(
        Rectangle(
            (lon250, y),
            lon500 - lon250,
            bar_h,
            facecolor="white",
            edgecolor="#556184",
            linewidth=0.9,
            zorder=20,
        )
    )

    label_y = y + 0.022 * h

    for xx, txt in [
        (lon0, "0"),
        (lon250, "250"),
        (lon500, "500 km"),
    ]:
        ax.text(
            xx,
            label_y,
            txt,
            ha="center",
            va="bottom",
            fontsize=SMALL_FS,
            family=FONT_FAMILY,
            color=TEXT,
            zorder=21,
        )


# ============================================================
# 10. MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # Downloads
    # --------------------------------------------------------
    download_if_missing(ADMIN1_URL, ADMIN1_ZIP)
    download_if_missing(ADMIN0_URL, ADMIN0_ZIP)

    if USE_RIVERS:
        download_if_missing(RIVERS_URL, RIVERS_ZIP)

    relief_tif, relief_tfw = ensure_relief_extracted()

    # --------------------------------------------------------
    # Read vector data
    # --------------------------------------------------------
    admin1 = gpd.read_file(f"zip://{ADMIN1_ZIP.resolve()}")
    admin0 = gpd.read_file(f"zip://{ADMIN0_ZIP.resolve()}")

    china0 = get_china_admin0(admin0).to_crs("EPSG:4326")
    china1 = get_china_admin1(admin1).to_crs("EPSG:4326")

    gd = find_province(china1, "Guangdong")
    gx = find_province(china1, "Guangxi")

    gd_geom = geom_union(gd)
    gx_geom = geom_union(gx)
    target = gd_geom.union(gx_geom)

    xmin, ymin, xmax, ymax = target.bounds
    w = xmax - xmin
    h = ymax - ymin

    # Main map extent.
    xlo = xmin - 0.10 * w
    xhi = xmax + 0.14 * w
    ylo = ymin - 0.20 * h
    yhi = ymax + 0.17 * h

    bbox_geom = box(xlo, ylo, xhi, yhi)

    # --------------------------------------------------------
    # Read rivers if enabled
    # --------------------------------------------------------
    rivers = None
    if USE_RIVERS:
        try:
            rivers = gpd.read_file(f"zip://{RIVERS_ZIP.resolve()}").to_crs("EPSG:4326")
        except Exception as exc:
            print("WARNING: could not read rivers:", exc)
            rivers = None

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    fig = plt.figure(
        figsize=FIGSIZE,
        facecolor="white",
    )

    ax = fig.add_axes([0.025, 0.055, 0.78, 0.90])
    ax.set_aspect("equal")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    # --------------------------------------------------------
    # Real shaded relief underlay
    # --------------------------------------------------------
    if USE_REAL_RELIEF and relief_tif and relief_tfw:
        try:
            relief, relief_extent = crop_relief(
                relief_tif,
                relief_tfw,
                xlo,
                xhi,
                ylo,
                yhi,
            )

            ax.imshow(
                relief,
                cmap="gray",
                vmin=0,
                vmax=1,
                extent=relief_extent,
                origin="upper",
                interpolation="bilinear",
                alpha=RELIEF_ALPHA,
                zorder=0,
            )

        except Exception as exc:
            print("WARNING: relief layer failed:", exc)

    # --------------------------------------------------------
    # Neighboring province fills and boundaries
    # --------------------------------------------------------
    for geom in china1.geometry:
        if geom is None or geom.is_empty or not geom.intersects(bbox_geom):
            continue

        ax.add_patch(
            geometry_patch(
                geom,
                facecolor=LAND,
                edgecolor=BACKGROUND_BOUNDARY,
                linewidth=0.95,
                alpha=0.78,
                zorder=2,
            )
        )

    # --------------------------------------------------------
    # Main province pale bases
    # --------------------------------------------------------
    ax.add_patch(
        geometry_patch(
            gx_geom,
            facecolor=GX_LIGHT,
            edgecolor="none",
            alpha=0.78,
            zorder=3,
        )
    )
    ax.add_patch(
        geometry_patch(
            gd_geom,
            facecolor=GD_LIGHT,
            edgecolor="none",
            alpha=0.78,
            zorder=3,
        )
    )

    # --------------------------------------------------------
    # Watercolor overlays
    # --------------------------------------------------------
    watercolor_fill(
        ax,
        gx_geom,
        GX_LIGHT,
        GX_MID,
        GX_DEEP,
        seed=27,
        resolution=690,
        alpha=0.83,
        zorder=5,
    )

    watercolor_fill(
        ax,
        gd_geom,
        GD_LIGHT,
        GD_MID,
        GD_DEEP,
        seed=73,
        resolution=690,
        alpha=0.83,
        zorder=5,
    )

    # --------------------------------------------------------
    # Rivers — subtle but real
    # --------------------------------------------------------
    if USE_RIVERS and rivers is not None:
        add_rivers(
            ax,
            rivers,
            bbox_geom,
            zorder=7,
        )

    # --------------------------------------------------------
    # High-contrast province outlines
    # --------------------------------------------------------
    ax.add_patch(
        geometry_patch(
            gx_geom,
            facecolor="none",
            edgecolor=MAIN_BOUNDARY,
            linewidth=1.55,
            zorder=9,
        )
    )

    ax.add_patch(
        geometry_patch(
            gd_geom,
            facecolor="none",
            edgecolor=MAIN_BOUNDARY,
            linewidth=1.55,
            zorder=9,
        )
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    gx_c = gx_geom.representative_point()
    gd_c = gd_geom.representative_point()

    gx_label_x = gx_c.x - 0.078 * w
    gx_label_y = gx_c.y + 0.042 * h

    gd_label_x = gd_c.x + 0.122 * w
    gd_label_y = gd_c.y - 0.006 * h

    text_halo = [
        pe.withStroke(
            linewidth=4.0,
            foreground="white",
            alpha=0.94,
        )
    ]

    ax.text(
        gx_label_x,
        gx_label_y,
        "Guangxi\n(region 2)",
        ha="center",
        va="center",
        fontsize=REGION_FS,
        family=FONT_FAMILY,
        fontweight="bold",
        color=TEXT,
        linespacing=1.02,
        path_effects=text_halo,
        zorder=20,
    )

    ax.text(
        gd_label_x,
        gd_label_y,
        "Guangdong\n(region 1)",
        ha="center",
        va="center",
        fontsize=REGION_FS,
        family=FONT_FAMILY,
        fontweight="bold",
        color=TEXT,
        linespacing=1.02,
        path_effects=text_halo,
        zorder=20,
    )

    # --------------------------------------------------------
    # Spillover arrows
    # --------------------------------------------------------
    arrow_cx = (gx_c.x + gd_c.x) / 2 + 0.008 * w
    arrow_cy = (gx_c.y + gd_c.y) / 2 - 0.074 * h

    add_spillover_arrows(
        ax,
        arrow_cx,
        arrow_cy,
        w,
        h,
    )

    ax.text(
        arrow_cx,
        arrow_cy + 0.220 * h,
        "Spillover\nbetween regions",
        ha="center",
        va="center",
        fontsize=SPILLOVER_FS,
        family=FONT_FAMILY,
        fontweight="bold",
        color=TEXT,
        linespacing=0.98,
        path_effects=text_halo,
        zorder=21,
    )

    # --------------------------------------------------------
    # North arrow + scale bar
    # --------------------------------------------------------
    north_x = xlo + 0.043 * (xhi - xlo)
    north_y = ylo + 0.080 * (yhi - ylo)

    add_north_arrow(
        ax,
        north_x,
        north_y,
        w,
        h,
    )

    scale_x = xlo + 0.030 * (xhi - xlo)
    scale_y = ylo + 0.018 * (yhi - ylo)

    add_scale_bar(
        ax,
        scale_x,
        scale_y,
        center_lat=(ymin + ymax) / 2,
        h=h,
    )

    ax.axis("off")

    # --------------------------------------------------------
    # China inset
    # --------------------------------------------------------
    ax_in = fig.add_axes([0.768, 0.665, 0.16, 0.225])
    ax_in.set_aspect("equal")

    china_geom = geom_union(china0)

    ax_in.add_patch(
        geometry_patch(
            china_geom,
            facecolor="#E7E7E7",
            edgecolor="#BEBEBE",
            linewidth=0.55,
            zorder=1,
        )
    )

    ax_in.add_patch(
        geometry_patch(
            gx_geom,
            facecolor="#8FC8A2",
            edgecolor="none",
            zorder=3,
        )
    )

    ax_in.add_patch(
        geometry_patch(
            gd_geom,
            facecolor="#86A5EA",
            edgecolor="none",
            zorder=3,
        )
    )

    cx0, cy0, cx1, cy1 = china_geom.bounds
    ax_in.set_xlim(cx0 - 1.0, cx1 + 1.0)
    ax_in.set_ylim(cy0 - 1.0, cy1 + 1.0)

    inset_halo = [
        pe.withStroke(
            linewidth=2.8,
            foreground="white",
            alpha=0.92,
        )
    ]

    ax_in.text(
        0.72,
        0.84,
        "China",
        transform=ax_in.transAxes,
        ha="center",
        va="center",
        fontsize=INSET_FS,
        family=FONT_FAMILY,
        fontweight="bold",
        color=TEXT,
        path_effects=inset_halo,
        zorder=5,
    )

    ax_in.axis("off")

    ax_in.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax_in.transAxes,
            facecolor="none",
            edgecolor="#BFC0C2",
            linewidth=0.9,
            zorder=10,
        )
    )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------
    for path, kwargs in [
        (OUT_PNG, dict(dpi=600)),
        (OUT_PDF, dict()),
        (OUT_SVG, dict()),
    ]:
        fig.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.012,
            facecolor="white",
            **kwargs,
        )

    plt.close(fig)

    # --------------------------------------------------------
    # Source note
    # --------------------------------------------------------
    today = str(np.datetime64("today"))

    source_note = f"""Map source declaration
======================

Figure:
Guangdong–Guangxi bidirectional spillover map.

Vector administrative boundary data:
Natural Earth, 1:10m Admin-1 States, Provinces
File: ne_10m_admin_1_states_provinces.zip
Source: {ADMIN1_URL}

Country inset data:
Natural Earth, 1:10m Admin-0 Countries
File: ne_10m_admin_0_countries.zip
Source: {ADMIN0_URL}

Shaded-relief background:
Natural Earth, 1:10m Shaded Relief Basic (SR_LR)
File: SR_LR.zip
Source: {RELIEF_URL}
Natural Earth describes this relief as being derived from downsampled
SRTM Plus elevation data and clipped to the Natural Earth coastline.

River layer:
Natural Earth, 1:10m Rivers + Lake Centerlines
File: ne_10m_rivers_lake_centerlines.zip
Source: {RIVERS_URL}

Access date:
{today}

Styling:
Guangdong and Guangxi were highlighted programmatically.
The watercolor/ink-wash overlays, labels, directional spillover arrows,
north arrow, scale bar, and final composition were generated by the
authors in Python.

Suggested manuscript wording:
"Administrative boundaries, shaded relief and river centerlines were
obtained from Natural Earth (1:10m). Guangdong and Guangxi were
highlighted, and the final map styling was generated by the authors
in Python."
"""

    with open(OUT_NOTE, "w", encoding="utf-8") as f:
        f.write(source_note)

    print("\nFinished.")
    print("Outputs:")
    print(" ", OUT_PNG)
    print(" ", OUT_PDF)
    print(" ", OUT_SVG)
    print(" ", OUT_NOTE)


if __name__ == "__main__":
    main()
