#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tactile_routeB_ams (Auto Top-N Colors + per-run parts folder)

Adds:
- --n_colors: pick top N dominant color regions automatically (default 4)
- --parts_root + --run_prefix: auto create new output folder each run (no overwrite)
- backward compatible: --parts_dir works as alias of --parts_root

Recommended AMS workflow:
  python3 tactile_routeB_ams_2.py input.png \
    --as_parts --fuse_pitch 0.25 --part_gap_mm 0.15 \
    --n_colors 4 \
    --parts_root parts_out --run_prefix cat \
    --out combined.stl --debug_dir debug
"""

import os
import math
import json
import argparse
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import cv2

from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.ops import unary_union, triangulate
from shapely.affinity import rotate
import trimesh


# -------------------------
# Helpers: run folder (no overwrite)
# -------------------------
def make_run_dir(root: str, prefix: str = "parts") -> str:
    os.makedirs(root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(root, f"{prefix}_{ts}")
    path = base
    i = 1
    while os.path.exists(path):
        path = f"{base}_{i:02d}"
        i += 1
    os.makedirs(path, exist_ok=True)
    return path


# -------------------------
# Binary ops
# -------------------------
def fill_holes(binary_u8: np.ndarray) -> np.ndarray:
    inv = (1 - binary_u8).astype(np.uint8) * 255
    h, w = inv.shape
    ff = inv.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 128)
    outside = (ff == 128)
    filled = binary_u8.copy()
    filled[(~outside) & (inv == 255)] = 1
    return filled


def largest_cc(binary_u8: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if num <= 1:
        return binary_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return (labels == idx).astype(np.uint8)


def keep_largest_cc(binary_u8: np.ndarray) -> np.ndarray:
    return largest_cc(binary_u8)


def mask_remove_small(binary_u8: np.ndarray, min_area_px: int) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    out = np.zeros_like(binary_u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            out[labels == i] = 1
    return out


# -------------------------
# Silhouette (foreground)
# -------------------------
def build_silhouette(
    img_bgr: np.ndarray,
    bg_thresh: float = 18.0,
    dilate_ksize: int = 7,
    dilate_iter: int = 2,
    open_iter: int = 1,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    patch = np.concatenate(
        [
            img_bgr[0:20, 0:20].reshape(-1, 3),
            img_bgr[0:20, w - 20 : w].reshape(-1, 3),
            img_bgr[h - 20 : h, 0:20].reshape(-1, 3),
            img_bgr[h - 20 : h, w - 20 : w].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(patch, axis=0)
    dist = np.linalg.norm(img_bgr.astype(np.float32) - bg.astype(np.float32), axis=2)

    seed = (dist > bg_thresh).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=open_iter)
    seed = largest_cc(seed)

    dil_kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
    dil = cv2.dilate(seed, dil_kernel, iterations=dilate_iter)

    sil = fill_holes(dil)
    sil = largest_cc(sil)
    return sil


def crop_to_mask(img: np.ndarray, mask: np.ndarray, pad_px: int = 25):
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    y0 = max(0, y0 - pad_px)
    y1 = min(mask.shape[0] - 1, y1 + pad_px)
    x0 = max(0, x0 - pad_px)
    x1 = min(mask.shape[1] - 1, x1 + pad_px)
    return img[y0 : y1 + 1, x0 : x1 + 1].copy(), mask[y0 : y1 + 1, x0 : x1 + 1].copy()


# -------------------------
# KMeans + tolerance merge (magic-wand-like)
# -------------------------
def kmeans_labels(lab_img: np.ndarray, mask_bool: np.ndarray, k: int = 16, attempts: int = 5):
    pts = lab_img[mask_bool].reshape(-1, 3).astype(np.float32)
    if len(pts) < 200:
        raise ValueError("Mask too small for kmeans.")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.15)
    _, labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    return labels.reshape(-1), centers, pts


def merge_centers_by_tol(centers: np.ndarray, tol: float):
    if tol <= 0:
        return [[i] for i in range(len(centers))]

    groups = []
    for i, c in enumerate(centers):
        placed = False
        for g in groups:
            rep = centers[g[0]]
            if np.linalg.norm(c - rep) <= tol:
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    return groups


def build_masks_from_groups(label_img: np.ndarray, groups):
    masks = []
    for g in groups:
        m = np.isin(label_img, np.array(g, dtype=np.int32)).astype(np.uint8)
        masks.append(m)
    return masks


# -------------------------
# mask -> shapely
# -------------------------
def mask_to_shapely(
    mask_u8: np.ndarray,
    scale: float,
    h_px: int,
    simplify_mm: float = 0.25,
    min_area_mm2: float = 2.0,
):
    m = (mask_u8.astype(np.uint8) * 255)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(cnts) == 0:
        return MultiPolygon([])

    hier = hier[0]
    polys = []

    for i, c in enumerate(cnts):
        if hier[i][3] != -1:
            continue
        if len(c) < 3:
            continue

        shell_px = c[:, 0, :]
        shell = [(float(x) * scale, float((h_px - 1 - y)) * scale) for x, y in shell_px]

        holes = []
        child = hier[i][2]
        while child != -1:
            hc = cnts[child][:, 0, :]
            if len(hc) >= 3:
                hole = [(float(x) * scale, float((h_px - 1 - y)) * scale) for x, y in hc]
                holes.append(hole)
            child = hier[child][0]

        poly = Polygon(shell, holes)
        if poly.is_empty:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue

        poly = poly.simplify(simplify_mm, preserve_topology=True)
        if not poly.is_empty and poly.area >= min_area_mm2:
            polys.append(poly)

    if not polys:
        return MultiPolygon([])
    geom = unary_union(polys)
    return geom if not geom.is_empty else MultiPolygon([])


# -------------------------
# triangulate + extrude
# -------------------------
def triangulate_polygon_to_mesh(poly: Polygon, height: float):
    tris = triangulate(poly)
    tris = [t for t in tris if t.representative_point().within(poly)]

    verts = []
    faces = []
    vmap = {}

    def add_pt(x, y):
        key = (round(float(x), 6), round(float(y), 6))
        if key in vmap:
            return vmap[key]
        idx = len(verts)
        verts.append([float(x), float(y)])
        vmap[key] = idx
        return idx

    for t in tris:
        coords = list(t.exterior.coords)[:-1]
        if len(coords) != 3:
            continue
        idxs = [add_pt(x, y) for x, y in coords]
        faces.append(idxs)

    if len(faces) == 0 or len(verts) < 3:
        return None

    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    return trimesh.creation.extrude_triangulation(v, f, height=height)


def extrude_geometry(geom, height: float, z0: float = 0.0):
    if geom is None or geom.is_empty:
        return None

    meshes = []
    if isinstance(geom, Polygon):
        geoms = [geom]
    else:
        geoms = list(getattr(geom, "geoms", [geom]))

    for g in geoms:
        if not isinstance(g, Polygon) or g.area <= 0:
            continue
        m = triangulate_polygon_to_mesh(g, height)
        if m is None:
            continue
        m.apply_translation([0.0, 0.0, float(z0)])
        meshes.append(m)

    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


# -------------------------
# procedural textures (continuous mapping)
# -------------------------
def stripes(poly, pitch: float, width: float, angle_deg: float = 0.0):
    minx, miny, maxx, maxy = poly.bounds
    diag = math.hypot(maxx - minx, maxy - miny)

    lines = []
    y = miny - diag
    while y < maxy + diag:
        lines.append(LineString([(minx - diag, y), (maxx + diag, y)]))
        y += pitch

    if angle_deg != 0.0:
        lines = [rotate(l, angle_deg, origin=poly.centroid, use_radians=False) for l in lines]

    ribbons = [l.buffer(width / 2.0, cap_style=2, join_style=2) for l in lines]
    return unary_union(ribbons).intersection(poly)


def dots(poly, pitch: float, radius: float):
    minx, miny, maxx, maxy = poly.bounds
    discs = []
    y = miny
    row = 0
    while y <= maxy:
        x = minx + (0.5 * pitch if (row % 2 == 1) else 0.0)
        while x <= maxx:
            discs.append(Point(x, y).buffer(radius))
            x += pitch
        y += pitch
        row += 1
    return unary_union(discs).intersection(poly)


def crosshatch(poly, pitch: float, width: float, angle_base: float = 0.0):
    return unary_union(
        [
            stripes(poly, pitch, width, angle_base + 45.0),
            stripes(poly, pitch, width, angle_base - 45.0),
        ]
    ).intersection(poly)


def texture_from_lab(geom, lab_center: np.ndarray, pitch_base: float, width_base: float):
    L = float(lab_center[0])
    a = float(lab_center[1]) - 128.0
    b = float(lab_center[2]) - 128.0
    chroma = math.hypot(a, b)
    hue = (math.degrees(math.atan2(b, a)) + 360.0) % 360.0

    tL = np.clip((L - 40.0) / 200.0, 0.0, 1.0)
    tC = np.clip(chroma / 80.0, 0.0, 1.0)

    pitch = pitch_base * (0.85 + 0.45 * tL)
    width = width_base * (0.9 + 0.5 * tC)

    if chroma < 14.0:
        return crosshatch(geom, pitch=pitch, width=width, angle_base=0.0), "crosshatch_gray"

    if 0 <= hue < 120:
        return stripes(geom, pitch=pitch, width=width, angle_deg=hue), "stripes"
    elif 120 <= hue < 240:
        radius = max(0.35 * width, 0.35)
        return dots(geom, pitch=pitch * 1.05, radius=radius), "dots"
    else:
        return crosshatch(geom, pitch=pitch * 0.95, width=width, angle_base=hue), "crosshatch"


# -------------------------
# fuse (voxel union)
# -------------------------
def fuse_mesh_voxel(mesh: trimesh.Trimesh, pitch_mm: float = 0.25) -> trimesh.Trimesh:
    try:
        vg = mesh.voxelized(pitch=pitch_mm)
        try:
            vg = vg.fill()  # better with scipy
        except Exception:
            pass
        fused = vg.marching_cubes
        fused.remove_duplicate_faces()
        fused.remove_degenerate_faces()
        fused.remove_unreferenced_vertices()
        fused.fix_normals()
        return fused
    except Exception:
        return mesh


# -------------------------
# main pipeline
# -------------------------
def build_tactile_mesh(
    img_path: str,
    target_width_mm: float = 160.0,
    k: int = 16,
    tol: float = 12.0,
    contiguous: bool = False,
    n_colors: int = 4,

    base_thick: float = 2.4,
    line_width_mm: float = 1.0,
    line_height: float = 3.0,

    tex_height: float = 0.55,
    pitch: float = 3.0,
    tex_width: float = 1.0,

    relief_min: float = 1.0,
    relief_max: float = 3.0,

    simplify_mm: float = 0.25,
    min_area_mm2: float = 2.0,
    bg_thresh: float = 18.0,

    as_parts: bool = False,
    fuse_pitch: float = 0.25,
    part_gap_mm: float = 0.0,

    debug_dir: Optional[str] = None,
):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(img_path)

    # alpha -> white bg
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        white = np.ones_like(bgr) * 255.0
        img = (bgr * alpha + white * (1 - alpha)).astype(np.uint8)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = img[:, :, :3]

    sil = build_silhouette(img, bg_thresh=bg_thresh)
    img_c, sil_c = crop_to_mask(img, sil, pad_px=25)

    h_px, w_px = sil_c.shape
    scale = target_width_mm / float(w_px)

    gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    line = ((gray < 60) & (sil_c == 1)).astype(np.uint8)

    desired_px = max(1, int(round(line_width_mm / scale)))
    ksz = desired_px * 2 + 1
    line = cv2.dilate(line, np.ones((ksz, ksz), np.uint8), iterations=1)

    fill_mask = ((sil_c == 1) & (line == 0))

    lab = cv2.cvtColor(img_c, cv2.COLOR_BGR2LAB)
    labels_1d, centers, _ = kmeans_labels(lab, fill_mask, k=k)

    label_img = np.full((h_px, w_px), -1, dtype=np.int32)
    idx = np.where(fill_mask.reshape(-1))[0]
    label_img.reshape(-1)[idx] = labels_1d

    groups = merge_centers_by_tol(centers, tol=tol)
    merged_masks = build_masks_from_groups(label_img, groups)

    min_area_px = max(20, int(round(min_area_mm2 / (scale * scale))))

    cleaned: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    for gi, (g, m) in enumerate(zip(groups, merged_masks)):
        if m.sum() == 0:
            continue
        m2 = mask_remove_small(m, min_area_px=min_area_px)
        m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        if contiguous and m2.sum() > 0:
            m2 = keep_largest_cc(m2)
        area = int(m2.sum())
        if area <= 0:
            continue
        center = centers[np.array(g)].mean(axis=0)
        cleaned.append((gi, area, center, m2))

    cleaned.sort(key=lambda x: x[1], reverse=True)
    top = cleaned[: max(1, int(n_colors))]
    if len(top) == 0:
        raise RuntimeError("No valid color regions found. Try increasing k or lowering min_area_mm2 / tol.")

    # relief height: darker -> higher
    Ls = np.array([float(t[2][0]) for t in top], dtype=np.float32)
    Lmin, Lmax = float(Ls.min()), float(Ls.max())
    denom = max(1e-6, (Lmax - Lmin))

    def height_from_L(L):
        t = (float(L) - Lmin) / denom  # bright=1
        return float(relief_min + (1.0 - t) * (relief_max - relief_min))

    base_poly = Polygon([(0, 0), (w_px * scale, 0), (w_px * scale, h_px * scale), (0, h_px * scale)])
    base_mesh = triangulate_polygon_to_mesh(base_poly, base_thick)

    parts = [base_mesh]
    named_parts: Dict[str, trimesh.Trimesh] = {"base": base_mesh}

    picked_info = []

    for rank, (_, area, center_lab, mask01) in enumerate(top):
        geom = mask_to_shapely(mask01, scale=scale, h_px=h_px, simplify_mm=simplify_mm, min_area_mm2=min_area_mm2)
        if geom is None or geom.is_empty:
            continue

        if part_gap_mm > 0:
            try:
                geom2 = geom.buffer(-part_gap_mm)
                if geom2.is_empty:
                    geom2 = geom
            except Exception:
                geom2 = geom
        else:
            geom2 = geom

        h_relief = height_from_L(center_lab[0])
        m_relief = extrude_geometry(geom2, height=h_relief, z0=base_thick)

        tex_geom, tex_name = texture_from_lab(geom2, center_lab, pitch_base=pitch, width_base=tex_width)
        m_tex = None
        if tex_geom is not None and (not tex_geom.is_empty):
            m_tex = extrude_geometry(tex_geom, height=tex_height, z0=base_thick + h_relief)

        part_name = f"color{rank+1}"
        if as_parts:
            combo = trimesh.util.concatenate([m for m in [m_relief, m_tex] if m is not None])
            combo = fuse_mesh_voxel(combo, pitch_mm=fuse_pitch)
            parts.append(combo)
            named_parts[part_name] = combo
        else:
            if m_relief is not None:
                parts.append(m_relief)
            if m_tex is not None:
                parts.append(m_tex)

        picked_info.append(
            {
                "part": part_name,
                "rank": rank + 1,
                "area_px": area,
                "lab_center": [float(center_lab[0]), float(center_lab[1]), float(center_lab[2])],
                "relief_mm": float(h_relief),
                "texture": tex_name,
            }
        )

    # line part
    line_geom = mask_to_shapely(line, scale=scale, h_px=h_px, simplify_mm=simplify_mm, min_area_mm2=1.0)
    if line_geom is not None and not line_geom.is_empty:
        m_line = extrude_geometry(line_geom, height=line_height, z0=base_thick)
        if m_line is not None:
            parts.append(m_line)
            named_parts["line"] = m_line

    mesh = trimesh.util.concatenate([p for p in parts if p is not None])

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "crop.png"), img_c)
        cv2.imwrite(os.path.join(debug_dir, "silhouette.png"), (sil_c * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(debug_dir, "line.png"), (line * 255).astype(np.uint8))

        vis = np.zeros_like(img_c)
        palette = [(255, 80, 80), (80, 255, 80), (80, 80, 255), (255, 200, 80), (200, 80, 255)]
        for rank, (_, _, _, m) in enumerate(top):
            col = palette[rank % len(palette)]
            for ch in range(3):
                vis[:, :, ch] = np.where(m == 1, col[ch], vis[:, :, ch])
        cv2.imwrite(os.path.join(debug_dir, "picked_top_colors.png"), vis)

        with open(os.path.join(debug_dir, "picked_info.json"), "w", encoding="utf-8") as f:
            json.dump(picked_info, f, ensure_ascii=False, indent=2)

    return mesh, named_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="input image path (jpg/png)")
    ap.add_argument("--out", default="tactile.stl", help="output combined STL path")
    ap.add_argument("--debug_dir", default=None, help="write debug pngs here")

    ap.add_argument("--target_width_mm", type=float, default=160.0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--tol", type=float, default=12.0)
    ap.add_argument("--contiguous", action="store_true")
    ap.add_argument("--n_colors", type=int, default=4, help="auto pick top N dominant colors by area")

    ap.add_argument("--base_thick", type=float, default=2.4)
    ap.add_argument("--line_width_mm", type=float, default=1.0)
    ap.add_argument("--line_height", type=float, default=3.0)

    ap.add_argument("--tex_height", type=float, default=0.55)
    ap.add_argument("--pitch", type=float, default=3.0)
    ap.add_argument("--tex_width", type=float, default=1.0)

    ap.add_argument("--relief_min", type=float, default=1.0)
    ap.add_argument("--relief_max", type=float, default=3.0)

    ap.add_argument("--simplify_mm", type=float, default=0.25)
    ap.add_argument("--min_area_mm2", type=float, default=2.0)
    ap.add_argument("--bg_thresh", type=float, default=18.0)

    ap.add_argument("--as_parts", action="store_true",
                    help="each picked color becomes ONE connected part (relief+texture fused)")
    ap.add_argument("--fuse_pitch", type=float, default=0.25)
    ap.add_argument("--part_gap_mm", type=float, default=0.0)

    # new: per-run folder (no overwrite)
    ap.add_argument("--parts_root", default=None,
                    help="Export per-part STLs into a NEW timestamped folder under this root.")
    ap.add_argument("--run_prefix", default="parts",
                    help="Prefix for the auto-created run folder name.")

    # backward-compatible alias
    ap.add_argument("--parts_dir", default=None,
                    help="Alias of --parts_root (kept for compatibility).")

    args = ap.parse_args()

    parts_root = args.parts_root if args.parts_root else args.parts_dir

    mesh, named_parts = build_tactile_mesh(
        args.image,
        target_width_mm=args.target_width_mm,
        k=args.k,
        tol=args.tol,
        contiguous=args.contiguous,
        n_colors=args.n_colors,
        base_thick=args.base_thick,
        line_width_mm=args.line_width_mm,
        line_height=args.line_height,
        tex_height=args.tex_height,
        pitch=args.pitch,
        tex_width=args.tex_width,
        relief_min=args.relief_min,
        relief_max=args.relief_max,
        simplify_mm=args.simplify_mm,
        min_area_mm2=args.min_area_mm2,
        bg_thresh=args.bg_thresh,
        as_parts=args.as_parts,
        fuse_pitch=args.fuse_pitch,
        part_gap_mm=args.part_gap_mm,
        debug_dir=args.debug_dir,
    )

    if parts_root:
        run_dir = make_run_dir(parts_root, prefix=args.run_prefix)
        for name, m in named_parts.items():
            if m is None:
                continue
            m.export(os.path.join(run_dir, f"{name}.stl"))
        print(f"Exported parts to: {run_dir}")

    mesh.export(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

