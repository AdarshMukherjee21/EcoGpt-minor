#!/usr/bin/env python3
"""Generate the ecoGPT icon set for the Open WebUI chat frontend.

Pure numpy + zlib (no new dependencies beyond the venv's numpy). The original
source assets lived in the removed vendor/dsh-harness tree; the generated
icons are already in place under the dashboard/openwebui statics and this
script is kept as the regeneration recipe. Historical output paths:
  vendor/dsh-harness/apps/web/public/favicon.svg         (light leaf, transparent)
  vendor/dsh-harness/apps/web/public/favicon-dark.svg    (dark-scheme leaf, transparent)
  vendor/dsh-harness/apps/web/public/apple-touch-icon.png (180x180 tile)
  vendor/dsh-harness/apps/web/public/icon-192.png
  vendor/dsh-harness/apps/web/public/icon-512.png
  vendor/dsh-harness/apps/web/public/icon-maskable-512.png

Run:  ./.venv/Scripts/python.exe tools/make_ecogpt_icons.py
"""
from __future__ import annotations

import io
import struct
import zlib
from pathlib import Path

import numpy as np

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "dsh-harness"
WEB_PUBLIC = VENDOR / "apps" / "web" / "public"
DESKTOP_RES = VENDOR / "apps" / "desktop" / "resources"

# Leaf geometry (same double cubic as packages/client/ui-primitives/src/FishLogo.tsx),
# normalized into a 23.16 x 17.04 box.
LEAF_BASE = (0.3, 16.0)
LEAF_TIP = (22.9, 0.2)
LEAF_UPPER = ((1.5, 8.0), (7.0, 1.2))    # cubic control points base -> tip
LEAF_LOWER = ((20.0, 12.4), (11.0, 17.2))  # cubic control points tip -> base (closed)

LEAF_GREEN_DARK_SCHEME = "#34D399"  # emerald-400
LEAF_GREEN = "#0E9F6E"              # emerald-600-ish for light favicon
TILE_TOP = (16, 185, 129)           # emerald-500
TILE_BOTTOM = (4, 120, 87)          # emerald-700
TILE_RADIUS_FRAC = 0.22


def _flatten(p0, c1, c2, p1, n: int = 96) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return ((1 - t) ** 3) * np.array(p0) + 3 * ((1 - t) ** 2) * t * np.array(c1) \
        + 3 * (1 - t) * (t ** 2) * np.array(c2) + (t ** 3) * np.array(p1)


def _leaf_polygon() -> np.ndarray:
    upper = _flatten(LEAF_BASE, *LEAF_UPPER, LEAF_TIP)
    lower = _flatten(LEAF_TIP, *LEAF_LOWER, LEAF_BASE)
    return np.vstack([upper, lower[1:-1]])  # ccw-ish closed ring


def _inside(px: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon; px shape (..., 2)."""
    x, y = px[..., 0], px[..., 1]
    inside = np.zeros(x.shape, dtype=bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for a, b, c, d in zip(x1, y1, x2, y2):
        crosses = ((b > y) != (d > y)) & (x < (c - a) * (y - b) / (d - b + 1e-12) + a)
        inside ^= crosses
    return inside


def _rounded_tile_mask(fx: np.ndarray, fy: np.ndarray, size: float, r: float) -> np.ndarray:
    inside = (fx >= 0) & (fx <= size) & (fy >= 0) & (fy <= size)
    r = min(r, size / 2)
    corners = [(0, 0), (size, 0), (0, size), (size, size)]
    for cx, cy in corners:
        in_corner_q = (fx < cx if cx == 0 else fx > cx) & (fy < cy if cy == 0 else fy > cy)
        dx, dy = fx - cx, fy - cy
        outside_round = in_corner_q & ((dx * dx + dy * dy) > r * r) & inside
        inside &= ~outside_round
    return inside


def render_tile(edge: int, maskable: bool = False, ss: int = 4) -> np.ndarray:
    """RGBA array of the app tile: rounded gradient tile + white leaf."""
    pad = 0.115 if not maskable else 0.24  # maskable keeps art inside the 80% safe zone
    mono = np.zeros((edge, edge, 4), dtype=np.uint8)
    # gradient per pixel row (background not subsampled — vertical, banding-free enough)
    g = np.linspace(0.0, 1.0, edge)[:, None]
    top = np.array(TILE_TOP, dtype=float)
    bot = np.array(TILE_BOTTOM, dtype=float)
    grad = (top * (1 - g[:, :, None]) + bot * g[:, :, None])

    # supersample the tile mask + leaf together
    off = (np.arange(ss) + 0.5) / ss - 0.5        # 1-D subpixel offsets
    xs = np.arange(edge)[:, None] + off[None, :]  # (edge, ss)
    ys = xs                                        # same offsets both axes
    fx = np.broadcast_to(xs[:, None, :, None], (edge, edge, ss, ss))
    fy = np.broadcast_to(ys[None, :, None, :], (edge, edge, ss, ss))
    fx = np.ascontiguousarray(fx)
    fy = np.ascontiguousarray(fy)

    r = edge * TILE_RADIUS_FRAC
    tile_ok = _rounded_tile_mask(fx, fy, float(edge), r)

    leaf = _leaf_polygon()
    lw, lh = 23.16, 17.04
    art_w = edge * (1 - 2 * pad)
    scale = art_w / lw
    art_h = lh * scale
    ox, oy = pad * edge, (edge - art_h) / 2
    # map supersampled tile pixels into leaf geometry space
    leaf_px = np.stack([(fx - ox) / scale, (fy - oy) / scale], axis=-1)
    leaf_ok = tile_ok & _inside(leaf_px, leaf)

    alpha = leaf_ok.mean(axis=(2, 3))
    mono[..., :3] = (grad.astype(np.uint8) * (1 - alpha[..., None]) + 255 * alpha[..., None]).astype(np.uint8)
    mono[..., 3] = 255
    return mono


def write_png(path: Path, rgba: np.ndarray) -> None:
    h, w = rgba.shape[:2]
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


LEAF_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 23.16 17.04" fill="none">'
    '<path d="M0.3 16.0 C1.5 8.0 7.0 1.2 22.9 0.2 C20.0 12.4 11.0 17.2 0.3 16.0 Z" fill="{color}"/>'
    "</svg>"
)

TILE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1104 1104" fill="none">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="rgb(16,185,129)"/><stop offset="1" stop-color="rgb(4,120,87)"/>'
    "</linearGradient></defs>"
    '<rect x="40" y="32" width="1024" height="1024" rx="225" fill="url(#g)"/>'
    '<g transform="translate({ox},{oy}) scale({s})">'
    '<path d="M0.3 16.0 C1.5 8.0 7.0 1.2 22.9 0.2 C20.0 12.4 11.0 17.2 0.3 16.0 Z" fill="#FFFFFF"/>'
    "</g></svg>"
)


def main() -> None:
    WEB_PUBLIC.mkdir(parents=True, exist_ok=True)
    DESKTOP_RES.mkdir(parents=True, exist_ok=True)

    (WEB_PUBLIC / "favicon.svg").write_text(LEAF_SVG.format(color=LEAF_GREEN), encoding="utf-8")
    (WEB_PUBLIC / "favicon-dark.svg").write_text(LEAF_SVG.format(color=LEAF_GREEN_DARK_SCHEME), encoding="utf-8")

    # Desktop tile source: 1104 square, tile inset 40/32 + edge 1024 (render-app-icons contract).
    inset_x, inset_y, tile_edge = 40, 32, 1024
    s = (tile_edge * (1 - 2 * 0.115)) / 23.16
    art_w = 23.16 * s
    ox = inset_x + (tile_edge - art_w) / 2
    oy = inset_y + (tile_edge - 17.04 * s) / 2
    (DESKTOP_RES / "icon.svg").write_text(
        TILE_SVG.format(ox=f"{ox:.1f}", oy=f"{oy:.1f}", s=f"{s:.4f}"), encoding="utf-8")

    write_png(WEB_PUBLIC / "apple-touch-icon.png", render_tile(180))
    write_png(WEB_PUBLIC / "icon-192.png", render_tile(192))
    write_png(WEB_PUBLIC / "icon-512.png", render_tile(512))
    write_png(WEB_PUBLIC / "icon-maskable-512.png", render_tile(512, maskable=True))
    print("ecoGPT icons written to", WEB_PUBLIC)


if __name__ == "__main__":
    main()
