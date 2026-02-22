#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_from_masks.py — Route A (mask-driven tactile bas-relief)

Inputs (same resolution as the original image):
  - mask_box.png    : white = box body
  - mask_fries.png  : white = fries (separated if you want "root-by-root")
  - crease_map.png  : white = crease/outline lines (ridge)

Outputs:
  - main STL (tactile bas-relief)
  - optional legend STL (3 swatches: smooth / ridge / rough)

Key idea:
  - Build a single height-map in mm (base thickness everywhere)
  - Add:
      box smooth arc (low-frequency bulge) + optional very slight arc
      crease ridges (thin, high-contrast)
      fries: higher body + controlled roughness (filtered noise)
  - Convert height-map to a watertight solid via trimesh.creation.heightfield()

Dependencies:
  pip install numpy opencv-python trimesh

Example (PowerShell, venv activated):
  python build_from_masks.py `
    --mask_box mask_box.png `
    --mask_fries mask_fries.png `
    --crease crease_map.png `
    --target_width_mm 160 `
    --out fries_main.stl `
    --legend_out fries_legend.stl
"""

import argparse
import os
import numpy as np
import cv2
import trimesh


# -------------------------
# Utility: load grayscale mask as float 0..1
# -------------------------
def load_mask01(path: str, thresh: int = 128) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img > thresh).astype(np.float32)


def load_gray01(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img.astype(np.float32) / 255.0)


def soften_mask(mask01: np.ndarray, blur_px: float = 0.8) -> np.ndarray:
    """Slight edge soften to avoid jagged STL walls; keeps binary-ish look."""
    if blur_px <= 0:
        return mask01
    m = cv2.GaussianBlur(mask01, (0, 0), blur_px)
    return np.clip(m, 0.0, 1.0)


def dilate_mask(mask01: np.ndarray, px: int = 1) -> np.ndarray:
    if px <= 0:
        return mask01
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    d = cv2.dilate((mask01 > 0.5).astype(np.uint8), k, iterations=1)
    return d.astype(np.float32)


def erode_mask(mask01: np.ndarray, px: int = 1) -> np.ndarray:
    if px <= 0:
        return mask01
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    e = cv2.erode((mask01 > 0.5).astype(np.uint8), k, iterations=1)
    return e.astype(np.float32)


# -------------------------
# Box arc (smoothness + curvature)
# -------------------------
def box_arc_map(box01: np.ndarray, bulge_mm: float = 1.4, y_taper: float = 0.25) -> np.ndarray:
    """
    Create a smooth bulge for the box region.

    - Main curvature: cylindrical bulge across X inside the box bounding box
    - Slight taper across Y (optional) to keep top/bottom from over-bulging
    """
    if bulge_mm <= 0:
        return np.zeros_like(box01, dtype=np.float32)

    ys, xs = np.where(box01 > 0.5)
    if len(xs) < 10:
        return np.zeros_like(box01, dtype=np.float32)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    h, w = box01.shape
    yy, xx = np.indices((h, w), dtype=np.float32)

    cx = (x0 + x1) * 0.5
    rx = max(1.0, (x1 - x0) * 0.5)

    # cylindrical bulge across x: peak at center
    nx = (xx - cx) / rx
    bulge = 1.0 - np.clip(nx * nx, 0.0, 1.0)  # parabola

    if y_taper > 0:
        cy = (y0 + y1) * 0.5
        ry = max(1.0, (y1 - y0) * 0.5)
        ny = (yy - cy) / ry
        taper = 1.0 - np.clip((ny * ny) * y_taper, 0.0, 1.0)
        bulge *= taper

    bulge *= box01
    bulge *= bulge_mm
    return bulge.astype(np.float32)


# -------------------------
# Fries roughness (controlled, tactile-readable)
# -------------------------
def filtered_noise(shape, sigma_px: float = 2.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.random(shape, dtype=np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma_px)
    # normalize to 0..1
    n = n - float(n.min())
    denom = max(1e-6, float(n.max()))
    n = n / denom
    return n.astype(np.float32)


def fries_rough_map(fries01: np.ndarray, rough_mm: float = 0.6, sigma_px: float = 2.2, seed: int = 1) -> np.ndarray:
    """
    Roughness only on fries area.
    Use low-ish frequency bumps (not dense crosshatch) to stay tactile-readable.
    """
    if rough_mm <= 0:
        return np.zeros_like(fries01, dtype=np.float32)
    n = filtered_noise(fries01.shape, sigma_px=sigma_px, seed=seed)
    return (n * fries01 * rough_mm).astype(np.float32)


# -------------------------
# Crease ridges (thin, high-contrast lines)
# -------------------------
def crease_ridge_map(crease01: np.ndarray, ridge_mm: float = 0.9, line_dilate_px: int = 2) -> np.ndarray:
    """
    Convert crease map into raised ridges.
    Dilate to ensure it prints as a physically trackable ridge.
    """
    if ridge_mm <= 0:
        return np.zeros_like(crease01, dtype=np.float32)
    c = dilate_mask(crease01, px=line_dilate_px)
    # keep ridge crisp
    c = np.clip(c, 0.0, 1.0)
    return (c * ridge_mm).astype(np.float32)


# -------------------------
# Heightfield -> watertight mesh
# -------------------------
def heightmap_to_mesh(height_mm: np.ndarray, pitch_mm: float) -> trimesh.Trimesh:
    """
    trimesh.creation.heightfield produces a watertight solid (sides + bottom at z=0).
    height_mm is absolute height in mm above bottom (z=0).
    """
    # Ensure float64 for trimesh stability
    h = height_mm.astype(np.float64)
    mesh = trimesh.creation.heightfield(h, pitch=(pitch_mm, pitch_mm))

    # Cleanups (version-safe)
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    for fn in ("remove_degenerate_faces", "remove_unreferenced_vertices", "fix_normals"):
        if hasattr(mesh, fn):
            try:
                getattr(mesh, fn)()
            except Exception:
                pass

    return mesh


# -------------------------
# Legend generator
# -------------------------
def build_legend_mesh(
    pitch_mm: float,
    base_thick: float = 2.4,
    swatch_w_mm: float = 40.0,
    swatch_h_mm: float = 28.0,
    gap_mm: float = 6.0,
    ridge_mm: float = 0.9,
    rough_mm: float = 0.6,
) -> trimesh.Trimesh:
    """
    3 swatches side-by-side:
      1) smooth (flat)
      2) ridge (parallel ridges)
      3) rough (filtered noise bumps)
    """
    total_w = swatch_w_mm * 3 + gap_mm * 4
    total_h = swatch_h_mm + gap_mm * 2

    # choose pixel dims based on pitch
    W = int(round(total_w / pitch_mm))
    H = int(round(total_h / pitch_mm))
    hm = np.full((H, W), base_thick, dtype=np.float32)

    def mm_to_px(mm): return int(round(mm / pitch_mm))

    pad = mm_to_px(gap_mm)
    sw_w = mm_to_px(swatch_w_mm)
    sw_h = mm_to_px(swatch_h_mm)
    y0 = pad
    y1 = min(H, y0 + sw_h)

    # swatch 1: smooth (do nothing)
    x1_0 = pad
    x1_1 = min(W, x1_0 + sw_w)

    # swatch 2: ridge
    x2_0 = x1_1 + pad
    x2_1 = min(W, x2_0 + sw_w)
    if x2_1 > x2_0 and y1 > y0:
        # add parallel ridges
        ridge_pitch_px = max(2, mm_to_px(4.5))
        ridge_width_px = max(1, mm_to_px(1.4))
        for x in range(x2_0, x2_1, ridge_pitch_px):
            xw0 = x
            xw1 = min(x2_1, x + ridge_width_px)
            hm[y0:y1, xw0:xw1] += ridge_mm

    # swatch 3: rough
    x3_0 = x2_1 + pad
    x3_1 = min(W, x3_0 + sw_w)
    if x3_1 > x3_0 and y1 > y0:
        n = filtered_noise((y1 - y0, x3_1 - x3_0), sigma_px=2.2, seed=123)
        hm[y0:y1, x3_0:x3_1] += n * rough_mm

    return heightmap_to_mesh(hm, pitch_mm=pitch_mm)


# -------------------------
# Main builder (Route A)
# -------------------------
def build_tactile_from_masks(
    mask_box_path: str,
    mask_fries_path: str,
    crease_path: str,
    target_width_mm: float = 160.0,
    base_thick: float = 2.4,

    # BOX
    box_base_mm: float = 1.0,     # flat lift of box above base
    box_bulge_mm: float = 0.8,    # extra smooth curvature
    box_edge_soften_px: float = 0.8,

    # CREASE
    crease_ridge_mm: float = 0.9,
    crease_dilate_px: int = 2,

    # FRIES
    fries_extra_mm: float = 2.8,   # extra height above box surface
    fries_edge_soften_px: float = 0.6,
    fries_rough_mm: float = 0.6,
    fries_rough_sigma_px: float = 2.2,
    seed: int = 7,

    # SAFETY
    clip_max_mm: float = 10.0,

    debug_dir: str | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    box01 = load_mask01(mask_box_path)
    fries01 = load_mask01(mask_fries_path)
    crease01 = load_mask01(crease_path)

    # Sanity: same shape
    if box01.shape != fries01.shape or box01.shape != crease01.shape:
        raise ValueError("mask_box, mask_fries, crease_map must have identical image dimensions.")

    h, w = box01.shape
    pitch_mm = target_width_mm / float(w)

    # soften edges slightly (but keep masks crisp)
    box01s = soften_mask(box01, blur_px=box_edge_soften_px)
    fries01s = soften_mask(fries01, blur_px=fries_edge_soften_px)
    crease01s = soften_mask(crease01, blur_px=0.0)  # keep ridges crisp

    # Base height everywhere
    hm = np.full((h, w), base_thick, dtype=np.float32)

    # Box: flat + smooth bulge
    hm += box01s * box_base_mm
    hm += box_arc_map((box01 > 0.5).astype(np.float32), bulge_mm=box_bulge_mm, y_taper=0.25)

    # Crease ridges: add on top of box (works even if crease spills slightly)
    hm += crease_ridge_map(crease01s, ridge_mm=crease_ridge_mm, line_dilate_px=crease_dilate_px)

    # Fries: ensure fries are clearly above box
    # Put fries on top of (base + box_base + box_bulge peak) by adding extra
    hm += fries01s * fries_extra_mm

    # Fries roughness (bumps)
    hm += fries_rough_map((fries01 > 0.5).astype(np.float32), rough_mm=fries_rough_mm,
                          sigma_px=fries_rough_sigma_px, seed=seed)

    hm = np.clip(hm, 0.0, clip_max_mm)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        # debug height visualization (0..255)
        hm_norm = (hm - hm.min()) / max(1e-6, (hm.max() - hm.min()))
        cv2.imwrite(os.path.join(debug_dir, "heightmap_preview.png"), (hm_norm * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(debug_dir, "box01.png"), (box01 * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(debug_dir, "fries01.png"), (fries01 * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(debug_dir, "crease01.png"), (crease01 * 255).astype(np.uint8))

    mesh = heightmap_to_mesh(hm, pitch_mm=pitch_mm)

    info = {
        "pitch_mm": pitch_mm,
        "base_thick_mm": base_thick,
        "box_base_mm": box_base_mm,
        "box_bulge_mm": box_bulge_mm,
        "crease_ridge_mm": crease_ridge_mm,
        "fries_extra_mm": fries_extra_mm,
        "fries_rough_mm": fries_rough_mm,
        "fries_rough_sigma_px": fries_rough_sigma_px,
    }
    return mesh, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_box", required=True, help="mask_box.png (white=box)")
    ap.add_argument("--mask_fries", required=True, help="mask_fries.png (white=fries)")
    ap.add_argument("--crease", required=True, help="crease_map.png (white=crease lines)")
    ap.add_argument("--target_width_mm", type=float, default=160.0)
    ap.add_argument("--out", default="tactile_main.stl")
    ap.add_argument("--legend_out", default=None, help="optional legend STL output path")
    ap.add_argument("--debug_dir", default=None)

    # heights
    ap.add_argument("--base_thick", type=float, default=2.4)

    ap.add_argument("--box_base_mm", type=float, default=1.0)
    ap.add_argument("--box_bulge_mm", type=float, default=0.8)
    ap.add_argument("--crease_ridge_mm", type=float, default=0.9)
    ap.add_argument("--crease_dilate_px", type=int, default=2)

    ap.add_argument("--fries_extra_mm", type=float, default=2.8)
    ap.add_argument("--fries_rough_mm", type=float, default=0.6)
    ap.add_argument("--fries_rough_sigma_px", type=float, default=2.2)
    ap.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()

    mesh, info = build_tactile_from_masks(
        mask_box_path=args.mask_box,
        mask_fries_path=args.mask_fries,
        crease_path=args.crease,
        target_width_mm=args.target_width_mm,
        base_thick=args.base_thick,
        box_base_mm=args.box_base_mm,
        box_bulge_mm=args.box_bulge_mm,
        crease_ridge_mm=args.crease_ridge_mm,
        crease_dilate_px=args.crease_dilate_px,
        fries_extra_mm=args.fries_extra_mm,
        fries_rough_mm=args.fries_rough_mm,
        fries_rough_sigma_px=args.fries_rough_sigma_px,
        seed=args.seed,
        debug_dir=args.debug_dir,
    )

    mesh.export(args.out)
    print(f"Saved main STL: {args.out}")
    print("Params:", info)

    if args.legend_out:
        pitch_mm = info["pitch_mm"]
        legend = build_legend_mesh(
            pitch_mm=pitch_mm,
            base_thick=args.base_thick,
            ridge_mm=args.crease_ridge_mm,
            rough_mm=args.fries_rough_mm,
        )
        legend.export(args.legend_out)
        print(f"Saved legend STL: {args.legend_out}")


if __name__ == "__main__":
    main()