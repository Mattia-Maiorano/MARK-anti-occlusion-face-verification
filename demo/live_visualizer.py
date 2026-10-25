"""
live_visualizer.py
------------------
M.A.R.K. Real-Time Surveillance HUD Demonstration.

Cycles through all probe images from the dataset (applying all four synthetic
occlusion conditions), overlaying a zone-level exposure status HUD and
biometric confidence tags on each face frame.

Display layout (640 × 480 canvas)
-----------------------------------
┌──────────────────┬─────────────────────┬────────────────────────┐
│  Zone exposure   │  Probe face (112px) │  Gallery face (112px)  │
│  status bars     │  + biometric score  │  + subject ID          │
│  (Green/Red)     │  + MATCH/NO MATCH   │                        │
├──────────────────┴─────────────────────┴────────────────────────┤
│  Weights bar  |  Entropy  |  Condition  |  Controls             │
└────────────────────────────────────────────────────────────────┘

Controls
--------
    SPACE / → : next frame / condition
    ←         : previous
    Q / ESC   : quit
    S         : save current HUD frame as PNG to results/

Usage
-----
    python demo/live_visualizer.py \\
        --model   models/face_recognizer_fast.onnx \\
        --gallery data/gallery \\
        --probes  data/probes  \\
        [--results results]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

# Project root on path so src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embedder             import FaceEmbedder
from src.spatial_slicer       import resize_to_canonical, slice_into_zones, ZONE_NAMES, ZONE_BOUNDS
from src.occlusion_estimator  import compute_exposure_vector
from src.occlusion_injector   import apply_occlusion, OCCLUSION_MODES
from src.dynamic_matcher      import match, STATUS_VALID, DISCRIMINABILITY_PRIORS

# ── HUD constants ─────────────────────────────────────────────────────────────
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_CANVAS_W   = 680
_CANVAS_H   = 480
_BG_COLOR   = (22, 22, 32)      # dark navy background
_ACCENT     = (100, 220, 255)   # cyan accent for M.A.R.K. branding
_OK_COLOR   = (60, 200, 60)     # green  → zone exposed
_BAD_COLOR  = (40, 40, 220)     # red    → zone occluded
_INDET_CLR  = (30, 165, 255)    # orange → indeterminate
_MATCH_CLR  = (50, 200, 50)
_NOMATCH_CLR= (50, 50, 210)
_WHITE      = (240, 240, 240)
_GRAY       = (120, 120, 120)
_YELLOW     = (30, 210, 210)

_FACE_X   = 185    # x origin of probe face panel
_FACE_Y   = 70
_GAL_X    = 390    # x origin of gallery face panel
_GAL_Y    = 70
_ZONE_X   = 18     # x origin of zone bar panel
_ZONE_Y   = 70


def _put(canvas, text, xy, scale=0.42, color=_WHITE, thickness=1):
    cv2.putText(canvas, text, xy, _FONT, scale, color, thickness, cv2.LINE_AA)


def _bar(canvas, xy1, xy2, color, filled=True):
    if filled:
        cv2.rectangle(canvas, xy1, xy2, color, -1)
    cv2.rectangle(canvas, xy1, xy2, (0, 0, 0), 1)


def _draw_zone_bars(canvas: np.ndarray, E: np.ndarray) -> None:
    """Draw 4 coloured zone status bars with exposure score label."""
    bar_h, bar_w, gap = 28, 150, 5
    for i, (name, E_i) in enumerate(zip(ZONE_NAMES, E)):
        color  = _OK_COLOR if E_i > 0.0 else _BAD_COLOR
        y      = _ZONE_Y + i * (bar_h + gap)
        _bar(canvas, (_ZONE_X, y), (_ZONE_X + bar_w, y + bar_h), color)
        label  = f"Z{i+1} {name[:9]:<9}  {E_i:.2f}"
        _put(canvas, label, (_ZONE_X + 5, y + 19), scale=0.38, color=(0, 0, 0), thickness=1)


def _draw_weight_bars(canvas: np.ndarray, weights: np.ndarray, y_base: int = 310) -> None:
    """Draw a 4-segment proportional weight strip."""
    total_w = 150
    x = _ZONE_X
    _put(canvas, "Fusion weights:", (_ZONE_X, y_base - 4), scale=0.36, color=_GRAY)
    colors = [(255, 140, 0), (0, 200, 255), (180, 120, 255), (0, 200, 150)]
    for i, (w, c) in enumerate(zip(weights, colors)):
        seg_w = int(round(float(w) * total_w))
        if seg_w > 0:
            _bar(canvas, (x, y_base), (x + seg_w, y_base + 14), c)
            if seg_w > 20:
                _put(canvas, f"Z{i+1}", (x + 2, y_base + 11), scale=0.30,
                     color=(0, 0, 0), thickness=1)
        x += seg_w


def _draw_face_panel(canvas, img_112, x, y, label, color):
    """Blit a 112×112 face image at (x, y) with a coloured border and label."""
    canvas[y:y+112, x:x+112] = img_112
    cv2.rectangle(canvas, (x-1, y-1), (x+112, y+112), color, 1)
    _put(canvas, label, (x, y - 7), scale=0.40, color=color)


def build_hud(
    probe_img:    np.ndarray,
    gallery_img:  np.ndarray,
    E:            np.ndarray,
    result:       dict,
    probe_pid:    str,
    gallery_pid:  str,
    condition:    str,
    frame_idx:    int,
    n_total:      int,
) -> np.ndarray:
    """Composite the full 680 × 480 HUD canvas for one frame."""
    canvas = np.full((_CANVAS_H, _CANVAS_W, 3), _BG_COLOR, dtype=np.uint8)

    # ── Title ─────────────────────────────────────────────────────────────────
    _put(canvas, "M.A.R.K.  Surveillance HUD",
         (10, 22), scale=0.65, color=_ACCENT, thickness=1)
    cond_label = condition.replace("_", " ")
    _put(canvas, f"Condition: {cond_label}   Frame {frame_idx+1}/{n_total}",
         (10, 44), scale=0.42, color=_YELLOW)

    # ── Zone exposure bars ────────────────────────────────────────────────────
    _draw_zone_bars(canvas, E)

    # ── Probe face ────────────────────────────────────────────────────────────
    _draw_face_panel(canvas, probe_img, _FACE_X, _FACE_Y,
                     f"PROBE: {probe_pid}", (140, 160, 255))

    # ── Gallery face ──────────────────────────────────────────────────────────
    _draw_face_panel(canvas, gallery_img, _GAL_X, _GAL_Y,
                     f"GALLERY: {gallery_pid}", (140, 255, 160))

    # ── Biometric result ──────────────────────────────────────────────────────
    status = result.get("status", "")
    dist   = result.get("distance")
    entropy= float(result.get("entropy", 0.0))
    weights= result.get("weights", np.zeros(4))

    rx, ry = _FACE_X, _FACE_Y + 120

    if status == STATUS_VALID and dist is not None:
        is_match = (dist < 0.40)
        verdict  = "MATCH" if is_match else "NO MATCH"
        v_color  = _MATCH_CLR if is_match else _NOMATCH_CLR
        _put(canvas, f"Dist:  {dist:.4f}", (rx, ry),       scale=0.44, color=_WHITE)
        _put(canvas, verdict,              (rx, ry + 18),   scale=0.55, color=v_color, thickness=1)
    else:
        _put(canvas, "INDETERMINATE",     (rx, ry),        scale=0.50, color=_INDET_CLR)
        _put(canvas, "Insufficient info", (rx, ry + 18),   scale=0.38, color=_GRAY)

    _put(canvas, f"Entropy: {entropy:.4f}", (rx, ry + 38), scale=0.38, color=_GRAY)

    # Discriminability prior labels
    _put(canvas, "Priors: " + "  ".join(f"D{i+1}={d:.2f}" for i, d in
         enumerate(DISCRIMINABILITY_PRIORS)), (_ZONE_X, 285),
         scale=0.32, color=_GRAY)

    # Weight bars
    if status == STATUS_VALID:
        _draw_weight_bars(canvas, weights, y_base=310)
    else:
        _put(canvas, "Weights: N/A (indeterminate)", (_ZONE_X, 320), scale=0.36, color=_GRAY)

    # ── Status legend ─────────────────────────────────────────────────────────
    _bar(canvas, (_ZONE_X, 340), (_ZONE_X + 14, 354), _OK_COLOR)
    _put(canvas, "= Zone exposed",  (_ZONE_X + 18, 352), scale=0.36, color=_GRAY)
    _bar(canvas, (_ZONE_X, 358), (_ZONE_X + 14, 372), _BAD_COLOR)
    _put(canvas, "= Zone occluded", (_ZONE_X + 18, 370), scale=0.36, color=_GRAY)

    # ── Condition colour strip ────────────────────────────────────────────────
    cond_colors = {
        "Baseline":        (60, 200, 60),
        "Lower_Occlusion": (200, 100, 40),
        "Upper_Occlusion": (200, 150, 30),
        "Dual_Occlusion":  (200, 50,  50),
    }
    strip_c = cond_colors.get(condition, (120, 120, 120))
    cv2.rectangle(canvas, (0, 0), (_CANVAS_W, 4), strip_c, -1)

    # ── Controls ──────────────────────────────────────────────────────────────
    _put(canvas, "SPACE/→ next  ← prev  Q/ESC quit  S save",
         (10, _CANVAS_H - 10), scale=0.38, color=_GRAY)

    return canvas


def run_demo(args: argparse.Namespace) -> None:
    """Main demo loop."""
    os.makedirs(args.results, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────────
    print("[demo] Loading SFace backbone...")
    embedder = FaceEmbedder(model_path=args.model)

    # ── Load gallery ───────────────────────────────────────────────────────────
    gallery_imgs = sorted(glob.glob(os.path.join(args.gallery, "*.jpg")))
    if not gallery_imgs:
        raise FileNotFoundError(f"No gallery images found in: {args.gallery}")

    gallery: dict[str, dict] = {}
    for path in gallery_imgs:
        pid = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is None:
            continue
        img_112 = resize_to_canonical(img)
        zones   = slice_into_zones(img_112)
        gallery[pid] = {
            "img_112":   img_112,
            "zone_vecs": embedder.embed_zones(zones),
        }
    print(f"[demo] Gallery loaded: {len(gallery)} subjects.")

    # ── Load probes ────────────────────────────────────────────────────────────
    probe_paths = sorted(glob.glob(os.path.join(args.probes, "*.jpg")))
    if not probe_paths:
        raise FileNotFoundError(f"No probe images found in: {args.probes}")

    # Build flat entry list: (probe_path, occlusion_mode)
    entries = [
        (path, mode)
        for path in probe_paths
        for mode in OCCLUSION_MODES
    ]
    n_total = len(entries)
    print(f"[demo] {n_total} probe × condition entries ready.  Press SPACE to advance.")

    # ── Window setup ──────────────────────────────────────────────────────────
    win = "M.A.R.K. Surveillance Demo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, _CANVAS_W, _CANVAS_H)

    idx = 0
    while True:
        probe_path, condition = entries[idx % n_total]

        # Resolve probe subject ID
        base      = os.path.splitext(os.path.basename(probe_path))[0]
        probe_pid = base.split("_frame_")[0] if "_frame_" in base else base

        raw = cv2.imread(probe_path)
        if raw is None:
            idx += 1
            continue

        probe_112 = resize_to_canonical(raw)
        occluded  = apply_occlusion(probe_112, condition)

        # Zone embeddings and exposure
        probe_zones = slice_into_zones(occluded)
        probe_vecs  = embedder.embed_zones(probe_zones)
        E           = compute_exposure_vector(probe_zones)

        # Select gallery counterpart (same subject if available, else first)
        gallery_pid  = probe_pid if probe_pid in gallery else next(iter(gallery))
        gdata        = gallery[gallery_pid]
        match_result = match(probe_vecs, gdata["zone_vecs"], E)

        hud = build_hud(
            probe_img   = occluded,
            gallery_img = gdata["img_112"],
            E           = E,
            result      = match_result,
            probe_pid   = probe_pid,
            gallery_pid = gallery_pid,
            condition   = condition,
            frame_idx   = idx % n_total,
            n_total     = n_total,
        )
        cv2.imshow(win, hud)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):          # Q or ESC → quit
            break
        elif key in (ord(" "), 83, 3, 0):  # SPACE, right-arrow → advance
            idx += 1
        elif key in (81, 2):               # left-arrow → back
            idx = max(0, idx - 1)
        elif key == ord("s"):              # S → save HUD frame
            out_path = os.path.join(args.results, f"hud_{idx:04d}_{condition}.png")
            cv2.imwrite(out_path, hud)
            print(f"[demo] Saved HUD → {out_path}")

    cv2.destroyAllWindows()
    print("[demo] Session ended.")


def main():
    parser = argparse.ArgumentParser(
        description="M.A.R.K. Surveillance HUD Demo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model",   required=True,        help="Path to SFace ONNX model.")
    parser.add_argument("--gallery", default="data/gallery", help="Gallery image directory.")
    parser.add_argument("--probes",  default="data/probes",  help="Probe image directory.")
    parser.add_argument("--results", default="results",      help="Directory for saved HUD frames.")
    run_demo(parser.parse_args())


if __name__ == "__main__":
    main()
