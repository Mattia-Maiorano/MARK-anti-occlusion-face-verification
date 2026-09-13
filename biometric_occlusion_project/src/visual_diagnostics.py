"""
visual_diagnostics.py
---------------------
Preprocessing Visual Diagnostic Pool Generator.

For a sample of enrolled subjects, renders a multi-column PNG grid that
verifies alignment, coordinate slicing, and synthetic occlusion masking do
not distort facial geometry.

Grid layout (7 columns × N_SUBJECTS rows per condition)
--------------------------------------------------------
  Col 1 : Original raw face crop at native resolution (loaded from disk)
  Col 2 : 112 × 112 canonical resize (SFace input)
  Col 3 : Synthetically occluded probe
  Col 4 : Zone 1 masked array  (Forehead)
  Col 5 : Zone 2 masked array  (Periocular)
  Col 6 : Zone 3 masked array  (Nose/Cheeks)
  Col 7 : Zone 4 masked array  (Mouth/Jaw)

Overlays
---------
  • Zone boundary lines (coloured horizontal rules on canonical image)
  • Exposure score  Eᵢ  text label on each zone image
  • Column headers and condition labels

Output
------
  results/debug_visual_pool/{subject_id}_{condition}.png

Third-party : OpenCV (imread, resize, putText, line), NumPy, Matplotlib
Custom      : Grid assembly, overlay rendering, zone-boundary annotation.
"""

from __future__ import annotations

import os
import random

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .spatial_slicer      import resize_to_canonical, slice_into_zones, ZONE_BOUNDS, ZONE_NAMES
from .occlusion_injector  import apply_all_conditions, OCCLUSION_MODES
from .occlusion_estimator import compute_exposure_vector, compute_exposure_score


# ── Layout constants ──────────────────────────────────────────────────────────
_THUMB_SIZE   = 112          # thumbnail height/width for grid cells
_N_COLS       = 7
_BORDER       = 4            # pixels between cells
_HEADER_H     = 28           # pixels for column header row
_LABEL_H      = 22           # pixels for label strip below each row

# Colours (BGR for OpenCV, then converted to RGB for display)
_ZONE_COLORS_BGR = [
    (255, 100, 50),   # Zone 1 Forehead  — blue-ish
    (50,  220, 50),   # Zone 2 Periocular — green
    (50,  150, 255),  # Zone 3 Nose/Mid  — orange
    (180, 50,  255),  # Zone 4 Mouth/Jaw — purple
]

_COL_LABELS = [
    "Raw Crop",
    "112×112 Canonical",
    "Occluded Probe",
    "Z1 Forehead",
    "Z2 Periocular",
    "Z3 Nose/Cheek",
    "Z4 Mouth/Jaw",
]


def _draw_zone_boundaries(img: np.ndarray) -> np.ndarray:
    """
    Draw coloured horizontal lines at zone boundaries on a 112×112 BGR image.
    Returns a copy with overlay.
    """
    out = img.copy()
    for i, (r_start, _) in enumerate(ZONE_BOUNDS):
        color = _ZONE_COLORS_BGR[i]
        cv2.line(out, (0, r_start), (111, r_start), color, 1)
    return out


def _draw_exposure_score(img: np.ndarray, score: float, zone_idx: int) -> np.ndarray:
    """Overlay exposure score text on a zone image."""
    out = img.copy()
    color = _ZONE_COLORS_BGR[zone_idx]
    label = f"E{zone_idx+1}={score:.3f}"
    # Shadow
    cv2.putText(out, label, (3, 109), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                (0, 0, 0), 2, cv2.LINE_AA)
    # Foreground
    cv2.putText(out, label, (3, 109), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                color, 1, cv2.LINE_AA)
    return out


def _text_img(text: str, w: int, h: int, font_scale: float = 0.30,
              bg: tuple = (30, 30, 30), fg: tuple = (220, 220, 220)) -> np.ndarray:
    """Create a small BGR label image with centred text."""
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    x = max(0, (w - tw) // 2)
    y = max(th, (h + th) // 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, fg, 1, cv2.LINE_AA)
    return img


def _resize_raw_crop(path: str) -> np.ndarray:
    """Load and resize raw crop to _THUMB_SIZE for display (aspect-preserved, padded)."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((_THUMB_SIZE, _THUMB_SIZE, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    scale = _THUMB_SIZE / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    # Pad to square
    canvas = np.zeros((_THUMB_SIZE, _THUMB_SIZE, 3), dtype=np.uint8)
    y_off = (_THUMB_SIZE - nh) // 2
    x_off = (_THUMB_SIZE - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas


def _build_row(
    raw_path:  str,
    canonical: np.ndarray,
    occluded:  np.ndarray,
    zones:     list[np.ndarray],
    exposure:  np.ndarray,
) -> np.ndarray:
    """
    Assemble a single 7-cell horizontal strip for one subject × condition.
    """
    S = _THUMB_SIZE

    # Col 1: raw crop
    c1 = _resize_raw_crop(raw_path)

    # Col 2: canonical with zone boundary lines
    c2 = _draw_zone_boundaries(canonical.copy())

    # Col 3: occluded probe
    c3 = occluded.copy()
    cv2.putText(c3, "OCCLUDED", (2, 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.28, (0, 255, 255), 1, cv2.LINE_AA)

    # Cols 4–7: masked zone images with exposure overlay
    zone_imgs = []
    for i, (z, e_i) in enumerate(zip(zones, exposure)):
        zi = _draw_exposure_score(z, float(e_i), i)
        zone_imgs.append(zi)

    cells = [c1, c2, c3] + zone_imgs

    # Apply thin white border to each cell
    bordered = []
    for cell in cells:
        b = cv2.copyMakeBorder(cell, _BORDER, _BORDER, _BORDER, _BORDER,
                               cv2.BORDER_CONSTANT, value=(80, 80, 80))
        bordered.append(b)

    row = np.concatenate(bordered, axis=1)
    return row


def generate_visual_pool(
    registry:    dict,
    embedder,
    results_dir: str = "results",
    n_subjects:  int = 5,
    seed:        int = 42,
) -> None:
    """
    Generate the preprocessing visual diagnostic pool.

    For each selected subject and each of the 4 occlusion conditions, saves
    a 7-column PNG grid to ``results/debug_visual_pool/``.

    Parameters
    ----------
    registry : dict
        Output of ``chokepoint_extractor.extract_chokepoint_dataset``.
    embedder : FaceEmbedder
        Initialised SFace backbone (only used for visual verification —
        embeddings are not saved here).
    results_dir : str
        Root output directory.
    n_subjects : int
        Number of distinct subjects to visualise.  Capped at available count.
    seed : int
        RNG seed for reproducible subject selection.
    """
    out_dir = os.path.join(results_dir, "debug_visual_pool")
    os.makedirs(out_dir, exist_ok=True)

    # Select subjects
    all_pids = sorted(registry.keys())
    random.seed(seed)
    selected = random.sample(all_pids, min(n_subjects, len(all_pids)))
    print(f"[visual_diagnostics] Generating pool for subjects: {selected}")

    S    = _THUMB_SIZE
    brd  = _BORDER
    cell_w = S + 2 * brd
    grid_w = cell_w * _N_COLS

    col_header_row = np.concatenate(
        [_text_img(lbl, cell_w, _HEADER_H) for lbl in _COL_LABELS],
        axis=1,
    )

    for pid in selected:
        entry       = registry[pid]
        gallery_path = entry["gallery_path"]
        probe_paths  = entry.get("probe_paths", [])

        if not probe_paths:
            print(f"  [WARN] Subject '{pid}': no probes, skipping.")
            continue

        # Use the first probe for diagnostics
        probe_path = probe_paths[0]

        # Load images
        raw_gallery = cv2.imread(gallery_path)
        raw_probe   = cv2.imread(probe_path)

        if raw_gallery is None or raw_probe is None:
            print(f"  [WARN] Subject '{pid}': image load failed, skipping.")
            continue

        gallery_112 = resize_to_canonical(raw_gallery)
        probe_112   = resize_to_canonical(raw_probe)

        all_conditions = apply_all_conditions(probe_112)

        for condition in OCCLUSION_MODES:
            occluded  = all_conditions[condition]
            zones     = slice_into_zones(occluded)
            exposure  = compute_exposure_vector(zones)

            row = _build_row(probe_path, probe_112, occluded, zones, exposure)

            # Stack: header + row
            grid = np.vstack([col_header_row, row])

            # Add condition label banner at bottom
            banner = _text_img(
                f"Subject: {pid}  |  Condition: {condition}  |  "
                f"E=[{exposure[0]:.2f}, {exposure[1]:.2f}, "
                f"{exposure[2]:.2f}, {exposure[3]:.2f}]",
                grid_w, _LABEL_H,
                font_scale=0.32,
                bg=(15, 15, 15),
                fg=(200, 200, 50),
            )
            grid = np.vstack([grid, banner])

            # Save — convert BGR→RGB for matplotlib, or just use cv2
            fname = os.path.join(out_dir, f"{pid}_{condition}.png")
            cv2.imwrite(fname, grid)
            print(f"  ✓ Saved: {fname}")

    print(f"[visual_diagnostics] Pool saved to: {out_dir}")
