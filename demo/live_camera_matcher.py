"""
live_camera_matcher.py
-----------------------
M.A.R.K. Live Camera & Reference Photo Face Matcher Demonstration.

Captures live video feed from webcam, matches detected faces against a reference
uploaded photo in real-time, and displays zone exposure status HUD alongside
comparative confidence scores for all three biometric systems:
  1. Holistic Model (Full-face embedding)
  2. Patch-Only Model (Static weighted 4-zone distance)
  3. Hybrid v6 Model (Dynamic score-level gated fusion + entropy validation)

Controls
--------
    O       : Open/select reference photo path
    C       : Cycle webcam input source ID (0, 1, 2...)
    S       : Save current HUD snapshot frame to results/
    Q / ESC : Quit live camera matcher

Usage
-----
    python demo/live_camera_matcher.py \
        --model  models/face_recognizer_fast.onnx \
        --photo  data/gallery/P1L_S1_C1.1_0.jpg \
        [--camera 0] [--results results]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

# Project root on path so src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embedder             import FaceEmbedder
from src.spatial_slicer       import resize_to_canonical, slice_into_zones, ZONE_NAMES
from src.occlusion_estimator  import compute_exposure_vector
from src.dynamic_matcher      import (
    cosine_distance,
    match,
    STATUS_VALID,
    DISCRIMINABILITY_PRIORS,
    PATCH_PRIORS,
)

# ── HUD constants ─────────────────────────────────────────────────────────────
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_CANVAS_W   = 780
_CANVAS_H   = 520
_BG_COLOR   = (20, 22, 30)      # dark navy background
_ACCENT     = (100, 220, 255)   # cyan accent for M.A.R.K. branding
_OK_COLOR   = (60, 200, 60)     # green  → zone exposed
_BAD_COLOR  = (40, 40, 220)     # red    → zone occluded
_INDET_CLR  = (30, 165, 255)    # orange → indeterminate
_MATCH_CLR  = (50, 220, 50)
_NOMATCH_CLR= (50, 50, 220)
_WHITE      = (240, 240, 240)
_GRAY       = (140, 140, 150)
_DARK_GRAY  = (40, 44, 55)
_YELLOW     = (30, 210, 210)

_REF_X   = 20
_REF_Y   = 70
_CAM_X   = 160
_CAM_Y   = 70
_HUD_X   = 490
_HUD_Y   = 70

MATCH_THRESHOLD = 0.40


def _put(canvas: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.42, color: tuple = _WHITE, thickness: int = 1) -> None:
    cv2.putText(canvas, text, xy, _FONT, scale, color, thickness, cv2.LINE_AA)


def _bar(canvas: np.ndarray, xy1: tuple[int, int], xy2: tuple[int, int], color: tuple, filled: bool = True) -> None:
    if filled:
        cv2.rectangle(canvas, xy1, xy2, color, -1)
    cv2.rectangle(canvas, xy1, xy2, (50, 55, 70), 1)


def build_live_hud(
    ref_img_112: np.ndarray,
    cam_display_crop: np.ndarray,
    E: np.ndarray,
    ref_name: str,
    fps: float,
    holistic_dist: float,
    patch_dist: float,
    hybrid_result: dict,
    n_genuine: int = 0,
    n_impostor: int = 0,
    flash_msg: str = "",
    flash_color: tuple = _WHITE,
) -> np.ndarray:
    """Composite the full 780 × 520 HUD canvas for live camera matching."""
    canvas = np.full((_CANVAS_H, _CANVAS_W, 3), _BG_COLOR, dtype=np.uint8)

    # ── Title & Status Header ──────────────────────────────────────────────────
    _put(canvas, "M.A.R.K.  Live Camera & Photo Matcher", (15, 25), scale=0.65, color=_ACCENT, thickness=1)
    _put(canvas, f"FPS: {fps:.1f}   |   Ref: {ref_name[:25]}", (15, 48), scale=0.40, color=_YELLOW)

    # ── Reference Photo Panel ──────────────────────────────────────────────────
    ref_display = cv2.resize(ref_img_112, (120, 120))
    canvas[_REF_Y:_REF_Y+120, _REF_X:_REF_X+120] = ref_display
    cv2.rectangle(canvas, (_REF_X-1, _REF_Y-1), (_REF_X+120, _REF_Y+120), (100, 180, 255), 1)
    _put(canvas, "REF PHOTO", (_REF_X, _REF_Y - 8), scale=0.40, color=(100, 180, 255))

    # ── Live Camera Feed Panel ─────────────────────────────────────────────────
    cam_resized = cv2.resize(cam_display_crop, (300, 300))
    canvas[_CAM_Y:_CAM_Y+300, _CAM_X:_CAM_X+300] = cam_resized
    cv2.rectangle(canvas, (_CAM_X-1, _CAM_Y-1), (_CAM_X+300, _CAM_Y+300), _ACCENT, 1)
    _put(canvas, "LIVE CAMERA FEED", (_CAM_X, _CAM_Y - 8), scale=0.40, color=_ACCENT)

    # Flash Banner overlay on lower camera panel if active
    if flash_msg:
        cv2.rectangle(canvas, (_CAM_X + 5, _CAM_Y + 265), (_CAM_X + 295, _CAM_Y + 295), (10, 12, 18), -1)
        cv2.rectangle(canvas, (_CAM_X + 5, _CAM_Y + 265), (_CAM_X + 295, _CAM_Y + 295), flash_color, 1)
        _put(canvas, flash_msg, (_CAM_X + 10, _CAM_Y + 285), scale=0.38, color=flash_color, thickness=1)

    # ── Zone Exposure Bar Panel ────────────────────────────────────────────────
    _put(canvas, "ZONE EXPOSURE", (_REF_X, 215), scale=0.42, color=_ACCENT)
    bar_h, bar_w, gap = 20, 120, 4
    for i, (name, E_i) in enumerate(zip(ZONE_NAMES, E)):
        color = _OK_COLOR if E_i > 0.0 else _BAD_COLOR
        y = 225 + i * (bar_h + gap)
        _bar(canvas, (_REF_X, y), (_REF_X + bar_w, y + bar_h), color)
        label = f"Z{i+1} {name[:6]} {E_i:.2f}"
        _put(canvas, label, (_REF_X + 4, y + 14), scale=0.34, color=(0, 0, 0), thickness=1)

    # ── Comparative Models Panel (Right Sidebar) ──────────────────────────────
    rx = _HUD_X
    _put(canvas, "SYSTEM CONFIDENCE SCORES", (rx, 40), scale=0.48, color=_ACCENT, thickness=1)

    # 1. Holistic Model
    _put(canvas, "1. Holistic Model (Full Face)", (rx, 70), scale=0.40, color=_WHITE, thickness=1)
    h_match = holistic_dist < MATCH_THRESHOLD
    h_verdict = "MATCH" if h_match else "NO MATCH"
    h_color = _MATCH_CLR if h_match else _NOMATCH_CLR
    _put(canvas, f"Distance : {holistic_dist:.4f}", (rx + 10, 90), scale=0.38, color=_GRAY)
    _put(canvas, f"Verdict  : {h_verdict}", (rx + 10, 108), scale=0.42, color=h_color, thickness=1)

    cv2.line(canvas, (rx, 122), (_CANVAS_W - 15, 122), _DARK_GRAY, 1)

    # 2. Patch-Only Model
    _put(canvas, "2. Patch-Only Model (Static)", (rx, 142), scale=0.40, color=_WHITE, thickness=1)
    p_match = patch_dist < MATCH_THRESHOLD
    p_verdict = "MATCH" if p_match else "NO MATCH"
    p_color = _MATCH_CLR if p_match else _NOMATCH_CLR
    _put(canvas, f"Distance : {patch_dist:.4f}", (rx + 10, 162), scale=0.38, color=_GRAY)
    _put(canvas, f"Verdict  : {p_verdict}", (rx + 10, 180), scale=0.42, color=p_color, thickness=1)

    cv2.line(canvas, (rx, 194), (_CANVAS_W - 15, 194), _DARK_GRAY, 1)

    # 3. Hybrid v6 Model (Dynamic Gated Fusion)
    _put(canvas, "3. Hybrid v6 (Gated Fusion)", (rx, 214), scale=0.42, color=_YELLOW, thickness=1)
    hy_status = hybrid_result.get("status", "")
    hy_dist = hybrid_result.get("distance")
    hy_entropy = float(hybrid_result.get("entropy", 0.0))
    hy_weights = hybrid_result.get("weights", np.zeros(4))

    if hy_status == STATUS_VALID and hy_dist is not None:
        hy_match = hy_dist < MATCH_THRESHOLD
        hy_verdict = "MATCH" if hy_match else "NO MATCH"
        hy_color = _MATCH_CLR if hy_match else _NOMATCH_CLR
        _put(canvas, f"Distance : {hy_dist:.4f}", (rx + 10, 234), scale=0.38, color=_WHITE)
        _put(canvas, f"Verdict  : {hy_verdict}", (rx + 10, 256), scale=0.52, color=hy_color, thickness=1)
    else:
        _put(canvas, "Verdict  : INDETERMINATE", (rx + 10, 234), scale=0.44, color=_INDET_CLR, thickness=1)
        _put(canvas, "(Insufficient exposure)", (rx + 10, 252), scale=0.35, color=_GRAY)

    _put(canvas, f"Entropy  : {hy_entropy:.4f}", (rx + 10, 274), scale=0.36, color=_GRAY)

    # Fusion weight distribution
    if hy_status == STATUS_VALID:
        _put(canvas, "Dynamic Weights:", (rx + 10, 296), scale=0.36, color=_GRAY)
        total_w = 200
        wx = rx + 10
        colors = [(255, 140, 0), (0, 200, 255), (180, 120, 255), (0, 200, 150)]
        for i, (w, c) in enumerate(zip(hy_weights, colors)):
            seg_w = int(round(float(w) * total_w))
            if seg_w > 0:
                _bar(canvas, (wx, 304), (wx + seg_w, 318), c)
                if seg_w > 18:
                    _put(canvas, f"Z{i+1}", (wx + 2, 315), scale=0.30, color=(0, 0, 0), thickness=1)
            wx += seg_w

    # ── Summary Match Comparison Box ─────────────────────────────────────────
    cv2.rectangle(canvas, (rx, 345), (_CANVAS_W - 15, 430), _DARK_GRAY, -1)
    cv2.rectangle(canvas, (rx, 345), (_CANVAS_W - 15, 430), _ACCENT, 1)

    _put(canvas, "LIVE VERDICT SUMMARY", (rx + 15, 365), scale=0.40, color=_ACCENT, thickness=1)
    if hy_status == STATUS_VALID and hy_dist is not None:
        final_text = "PASSED MATCH" if hy_dist < MATCH_THRESHOLD else "REJECTED MATCH"
        final_clr = _MATCH_CLR if hy_dist < MATCH_THRESHOLD else _NOMATCH_CLR
    else:
        final_text = "OCCLUSION BLOCKED"
        final_clr = _INDET_CLR

    _put(canvas, final_text, (rx + 15, 400), scale=0.55, color=final_clr, thickness=1)

    # ── Calibration Live Counter Box ─────────────────────────────────────────
    _put(canvas, f"CALIB SAMPLES | Gen: {n_genuine}  Imp: {n_impostor}  Tot: {n_genuine + n_impostor}",
         (rx, 455), scale=0.38, color=_ACCENT, thickness=1)

    # ── Footer & Controls ──────────────────────────────────────────────────────
    _put(canvas, "O: Photo | G: Log Match(1) | I: Log Impostor(0) | C: Summary | V: Cam | S: Save | Q: Quit",
         (15, _CANVAS_H - 12), scale=0.36, color=_GRAY)

    return canvas


def align_eyes_only(img: np.ndarray, left_eye: tuple[float, float], right_eye: tuple[float, float], target_size: tuple[int, int] = (112, 112)) -> np.ndarray:
    """
    Perform similarity transformation to align face strictly based on eye coordinates.
    Protects against landmark warp distortion under heavy lower face occlusion (e.g. masks, hand over mouth).
    """
    desired_left_eye = (0.3155605 * target_size[0], 0.4615741 * target_size[1])
    dY = float(right_eye[1] - left_eye[1])
    dX = float(right_eye[0] - left_eye[0])
    angle = float(np.degrees(np.arctan2(dY, dX)))
    dist = max(1e-6, float(np.hypot(dX, dY)))
    desired_dist = (1.0 - 2.0 * 0.3155605) * target_size[0]
    scale = float(desired_dist / dist)

    eyes_center = (float((left_eye[0] + right_eye[0]) / 2.0), float((left_eye[1] + right_eye[1]) / 2.0))
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    tX = float(target_size[0] * 0.5 - eyes_center[0])
    tY = float(desired_left_eye[1] - eyes_center[1])
    M[0, 2] += tX
    M[1, 2] += tY

    return cv2.warpAffine(img, M, target_size, flags=cv2.INTER_LINEAR)


def align_face_safely(img: np.ndarray, face: np.ndarray, recognizer: cv2.FaceRecognizerSF) -> np.ndarray:
    """
    Align face using SFace recognizer.alignCrop, with automatic fallback to eye-only alignment
    if landmark distortion or crop degradation occurs.
    """
    # Extract 5 landmarks: left_eye (4,5), right_eye (6,7), nose (8,9), r_mouth (10,11), l_mouth (12,13)
    left_eye = (float(face[4]), float(face[5]))
    right_eye = (float(face[6]), float(face[7]))

    # Perform standard 5-point alignment via FaceRecognizerSF
    try:
        aligned = recognizer.alignCrop(img, face)
        if aligned is not None and aligned.shape == (112, 112, 3):
            return aligned
    except Exception:
        pass

    # Fallback to robust eyes-only alignment matrix
    return align_eyes_only(img, left_eye, right_eye)


def load_reference(embedder: FaceEmbedder, detector: cv2.FaceDetectorYN, photo_path: str) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, str]:
    """Load and process reference photo into 112×112 canonical aligned crop, zone vectors, and holistic vector."""
    if not os.path.isfile(photo_path):
        raise FileNotFoundError(f"Reference photo not found at: {photo_path}")

    raw = cv2.imread(photo_path)  # cv2.imread loads image as BGR uint8
    if raw is None:
        raise ValueError(f"Failed to decode reference image file: {photo_path}")

    rh, rw = raw.shape[:2]
    detector.setInputSize((rw, rh))
    _, faces = detector.detect(raw)

    if faces is not None and len(faces) > 0:
        best_face = max(faces, key=lambda f: f[2] * f[3])
        ref_112 = align_face_safely(raw, best_face, embedder._recognizer)
    else:
        print("[live_matcher] Warning: No face detected in reference photo via YuNet. Falling back to direct canonical resize.")
        ref_112 = resize_to_canonical(raw)

    ref_zones = slice_into_zones(ref_112)
    ref_zone_vecs = embedder.embed_zones(ref_zones)
    ref_holistic_vec = embedder.embed_holistic(ref_112)
    ref_name = os.path.basename(photo_path)

    return ref_112, ref_zone_vecs, ref_holistic_vec, ref_name


def run_live_camera_demo(args: argparse.Namespace) -> None:
    """Main live camera & photo matcher execution loop."""
    os.makedirs(args.results, exist_ok=True)

    print(f"[live_matcher] Loading SFace backbone from {args.model}...")
    embedder = FaceEmbedder(model_path=args.model)

    # Check for calibrated parameters auto-loading
    from src.dynamic_matcher import get_default_v6_model
    v6_master = get_default_v6_model()
    if os.path.isfile(args.calibrated_scaler):
        v6_master.load_calibrated_scaler(args.calibrated_scaler)
    if os.path.isfile(args.calibrated_gating):
        v6_master.load_calibrated_gating_weights(args.calibrated_gating)

    # Initialize CSV logger file and existing sample counts
    csv_path = args.csv
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    genuine_count = 0
    impostor_count = 0

    if os.path.isfile(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 10:
                        lbl = int(float(parts[9]))
                        if lbl == 1:
                            genuine_count += 1
                        elif lbl == 0:
                            impostor_count += 1
            print(f"[live_matcher] Found existing dataset at {csv_path} (Genuine: {genuine_count}, Impostor: {impostor_count}).")
        except Exception as exc:
            print(f"[live_matcher] Warning reading calibration CSV counts: {exc}")

    # Initialize FaceDetectorYN (YuNet) for canonical landmark alignment
    yunet_model = args.detector
    if not os.path.isfile(yunet_model):
        fallback = os.path.join(os.path.dirname(__file__), "..", "models", "face_detection_yunet_2023mar.onnx")
        if os.path.isfile(fallback):
            yunet_model = fallback
        else:
            raise FileNotFoundError(f"YuNet face detector model not found at '{yunet_model}' or '{fallback}'.")

    detector_score_thresh = float(args.score_thresh)
    print(f"[live_matcher] Loading YuNet face detector from {yunet_model} (score_threshold={detector_score_thresh})...")
    detector = cv2.FaceDetectorYN.create(yunet_model, "", (300, 300), score_threshold=detector_score_thresh)

    print(f"[live_matcher] Loading initial reference photo: {args.photo}")
    ref_112, ref_zone_vecs, ref_holistic_vec, ref_name = load_reference(embedder, detector, args.photo)

    camera_id = args.camera
    print(f"[live_matcher] Opening WebCam (device index: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[live_matcher] Warning: Unable to open webcam {camera_id}. Attempting default index 0...")
        camera_id = 0
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {camera_id}. Please check camera connection/permissions.")

    win_name = "M.A.R.K. Live Camera Face Matcher"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, _CANVAS_W, _CANVAS_H)

    if args.show_canonical:
        cv2.namedWindow("Canonical Crop", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Canonical Crop", 224, 224)

    prev_time = time.time()
    fps = 0.0

    last_valid_face = None
    frames_missing = 0
    MAX_OCCLUSION_HOLD = int(args.max_hold)

    last_match_time = 0.0
    interval = max(0.01, float(args.interval))

    holistic_dist = 1.0
    patch_dist = 1.0
    hybrid_result: dict = {}
    E = np.ones(4, dtype=np.float32)

    # Flash Banner state
    flash_msg = ""
    flash_color = _WHITE
    flash_until = 0.0

    print(f"[live_matcher] Live loop active. Press 'G': Genuine, 'I': Impostor, 'C': Summary, 'Q': Quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[live_matcher] Frame capture failed. Re-trying...")
            time.sleep(0.05)
            continue

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, curr_time - prev_time))
        prev_time = curr_time

        display_frame = frame.copy()

        # Face detection & alignment via cv2.FaceDetectorYN
        fh, fw = frame.shape[:2]
        detector.setInputSize((fw, fh))
        _, faces = detector.detect(frame)

        if faces is not None and len(faces) > 0:
            best_face = max(faces, key=lambda f: f[2] * f[3])
            last_valid_face = best_face
            frames_missing = 0
        elif last_valid_face is not None and frames_missing < MAX_OCCLUSION_HOLD:
            best_face = last_valid_face
            frames_missing += 1
        else:
            best_face = None

        if best_face is not None:
            x, y, w, h = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
            cam_112 = align_face_safely(frame, best_face, embedder._recognizer)
            box_clr = _ACCENT if frames_missing == 0 else _INDET_CLR
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_clr, 2)
            if frames_missing > 0:
                _put(display_frame, f"TRACK HOLD ({frames_missing}/{MAX_OCCLUSION_HOLD})", (x, max(15, y - 8)), scale=0.45, color=_INDET_CLR, thickness=2)
        else:
            h, w = frame.shape[:2]
            sz = min(h, w)
            cy, cx = h // 2, w // 2
            cam_crop = frame[cy - sz // 2 : cy + sz // 2, cx - sz // 2 : cx + sz // 2]
            cam_112 = resize_to_canonical(cam_crop)

        if args.show_canonical:
            cv2.imshow("Canonical Crop", cv2.resize(cam_112, (224, 224)))

        dh, dw = display_frame.shape[:2]
        dsz = min(dh, dw)
        dcy, dcx = dh // 2, dw // 2
        cam_display_square = display_frame[dcy - dsz // 2 : dcy + dsz // 2, dcx - dsz // 2 : dcx + dsz // 2]

        # Compute Embeddings & Distances at specified interval
        if curr_time - last_match_time >= interval:
            last_match_time = curr_time

            cam_zones = slice_into_zones(cam_112)
            cam_zone_vecs = embedder.embed_zones(cam_zones)
            cam_holistic_vec = embedder.embed_holistic(cam_112)
            E = compute_exposure_vector(cam_zones)

            holistic_dist = cosine_distance(cam_holistic_vec, ref_holistic_vec)

            zone_dists = [cosine_distance(cam_zone_vecs[i], ref_zone_vecs[i]) for i in range(4)]
            valid_mask = (E > 0.0)
            if np.any(valid_mask):
                denom = float(np.sum(DISCRIMINABILITY_PRIORS[valid_mask]))
                patch_dist = float(np.sum(DISCRIMINABILITY_PRIORS[valid_mask] * np.array(zone_dists)[valid_mask])) / max(1e-6, denom)
            else:
                patch_dist = 1.0

            hybrid_result = match(
                probe_zones=cam_zone_vecs,
                probe_holistic=cam_holistic_vec,
                probe_exposure=E,
                gallery_zones=ref_zone_vecs,
                gallery_holistic=ref_holistic_vec,
            )

        # Clear expired flash message
        if curr_time > flash_until:
            flash_msg = ""

        # Render HUD
        hud_frame = build_live_hud(
            ref_img_112=ref_112,
            cam_display_crop=cam_display_square,
            E=E,
            ref_name=ref_name,
            fps=fps,
            holistic_dist=holistic_dist,
            patch_dist=patch_dist,
            hybrid_result=hybrid_result,
            n_genuine=genuine_count,
            n_impostor=impostor_count,
            flash_msg=flash_msg,
            flash_color=flash_color,
        )

        cv2.imshow(win_name, hud_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # Q or ESC -> exit
            break

        elif key in (ord("g"), ord("i")):  # G -> Genuine (1), I -> Impostor (0)
            label = 1 if key == ord("g") else 0
            if best_face is None or frames_missing > 0:
                print(f"[live_matcher] Data log REJECTED: No active valid face detection.")
                flash_msg = "LOG REJECTED: NO VALID FACE DETECTED"
                flash_color = _INDET_CLR
                flash_until = curr_time + 1.8
            else:
                c_zones = slice_into_zones(cam_112)
                c_zone_vecs = embedder.embed_zones(c_zones)
                c_holistic_vec = embedder.embed_holistic(cam_112)

                norm_hol = float(np.linalg.norm(c_holistic_vec))
                if norm_hol < 1e-6 or any(float(np.linalg.norm(z)) < 1e-6 for z in c_zone_vecs):
                    print(f"[live_matcher] Data log REJECTED: Zero embedding vector encountered.")
                    flash_msg = "LOG REJECTED: ZERO VECTOR"
                    flash_color = _INDET_CLR
                    flash_until = curr_time + 1.8
                else:
                    d1 = cosine_distance(c_zone_vecs[0], ref_zone_vecs[0])
                    d2 = cosine_distance(c_zone_vecs[1], ref_zone_vecs[1])
                    d3 = cosine_distance(c_zone_vecs[2], ref_zone_vecs[2])
                    d4 = cosine_distance(c_zone_vecs[3], ref_zone_vecs[3])
                    d_hol = cosine_distance(c_holistic_vec, ref_holistic_vec)
                    c_exp = compute_exposure_vector(c_zones)

                    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
                    with open(csv_path, "a", encoding="utf-8") as f:
                        if write_header:
                            f.write("d_z1,d_z2,d_z3,d_z4,d_holistic,E_1,E_2,E_3,E_4,label\n")
                        row_str = f"{d1:.6f},{d2:.6f},{d3:.6f},{d4:.6f},{d_hol:.6f},{c_exp[0]:.6f},{c_exp[1]:.6f},{c_exp[2]:.6f},{c_exp[3]:.6f},{label}\n"
                        f.write(row_str)
                        f.flush()

                    if label == 1:
                        genuine_count += 1
                        flash_msg = "LOGGED GENUINE SAMPLE (LABEL 1)"
                        flash_color = _MATCH_CLR
                        print(f"[live_matcher] ✓ Logged Genuine sample -> {csv_path} (Total Genuine: {genuine_count})")
                    else:
                        impostor_count += 1
                        flash_msg = "LOGGED IMPOSTOR SAMPLE (LABEL 0)"
                        flash_color = _NOMATCH_CLR
                        print(f"[live_matcher] ✓ Logged Impostor sample -> {csv_path} (Total Impostor: {impostor_count})")

                    flash_until = curr_time + 1.8

        elif key == ord("c"):  # C -> Print sample balance summary
            tot = genuine_count + impostor_count
            print("\n" + "=" * 54)
            print(" M.A.R.K. In-Situ Calibration Dataset Summary")
            print("=" * 54)
            print(f" CSV File Path    : {os.path.abspath(csv_path)}")
            print(f" Genuine (1)      : {genuine_count}")
            print(f" Impostor (0)     : {impostor_count}")
            print(f" Total Samples    : {tot}")
            print(f" Impostor Ratio   : {(impostor_count / max(1, tot)) * 100:.1f}%")
            print("=" * 54 + "\n")
            flash_msg = f"SUMMARY PRINTED (GEN: {genuine_count}, IMP: {impostor_count})"
            flash_color = _YELLOW
            flash_until = curr_time + 2.0

        elif key == ord("s"):  # S -> save HUD frame
            out_path = os.path.join(args.results, f"live_hud_{int(time.time())}.png")
            cv2.imwrite(out_path, hud_frame)
            print(f"[live_matcher] Saved HUD snapshot -> {out_path}")

        elif key in (ord("v"), ord("w")):  # V/W -> switch camera index
            camera_id = (camera_id + 1) % 3
            print(f"[live_matcher] Switching to camera ID: {camera_id}")
            cap.release()
            cap = cv2.VideoCapture(camera_id)

        elif key == ord("o"):  # O -> prompt for photo path in terminal
            print("\n[live_matcher] Enter new reference photo path:")
            new_path = input("Path > ").strip().strip("'\"")
            if os.path.isfile(new_path):
                try:
                    ref_112, ref_zone_vecs, ref_holistic_vec, ref_name = load_reference(embedder, detector, new_path)
                    print(f"[live_matcher] Reference photo updated: {new_path}")
                    last_match_time = 0.0
                except Exception as exc:
                    print(f"[live_matcher] Error loading reference photo: {exc}")
            else:
                print(f"[live_matcher] Invalid file path: {new_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[live_matcher] Session ended.")


def main():
    parser = argparse.ArgumentParser(
        description="M.A.R.K. Live Camera & Reference Photo Face Matcher",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to SFace ONNX model (face_recognizer_fast.onnx).")
    parser.add_argument("--detector", default="models/face_detection_yunet_2023mar.onnx", help="Path to YuNet face detector ONNX model.")
    parser.add_argument("--photo", default="data/gallery/P1L_S1_C1.1_0.jpg", help="Path to reference photo image.")
    parser.add_argument("--camera", type=int, default=0, help="WebCam device index (default: 0).")
    parser.add_argument("--score-thresh", type=float, default=0.3, help="Score threshold for YuNet live face detector (default: 0.3).")
    parser.add_argument("--max-hold", type=int, default=10, help="Maximum missing frames to hold previous valid face detection (default: 10).")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval in seconds between biometric matching score updates (default: 1.0).")
    parser.add_argument("--csv", default="data/calibration_samples.csv", help="Path to in-situ calibration CSV sample log file.")
    parser.add_argument("--calibrated-scaler", default="models/distance_scaler_calibrated.json", help="Path to calibrated DistanceScaler JSON file.")
    parser.add_argument("--calibrated-gating", default="models/gating_v6_calibrated.pth", help="Path to calibrated LearnedGatingNetwork weights.")
    parser.add_argument("--results", default="results", help="Directory for saved HUD snapshots.")
    parser.add_argument("--show-canonical", action="store_true", help="Display separate 'Canonical Crop' preview window.")
    run_live_camera_demo(parser.parse_args())


if __name__ == "__main__":
    main()

