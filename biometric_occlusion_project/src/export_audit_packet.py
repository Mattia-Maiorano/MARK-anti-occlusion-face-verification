#!/usr/bin/env python3
"""
export_audit_packet.py
----------------------
Export an audit evaluation packet for an external analyzer to review
the biometric verification software quality (M.A.R.K. v4 Architecture).

Outputs generated into the target export folder (default: 'audit_packet/'):
  1. 01_correct_matching.png  (True Accept / TA)
  2. 02_correct_refused.png   (True Reject / TR)
  3. 03_wrongly_matching.png  (False Accept / FA)
  4. 04_wrongly_refused.png   (False Reject / FR)
  5. formulas_and_weights.md  (Complete mathematical formulations, priors, weights, parameters)
  6. metadata.json            (Machine-readable specs of the sample cases, scores, and weights)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Any, List

import cv2
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.embedder import FaceEmbedder
from src.spatial_slicer import resize_to_canonical, slice_into_zones, ZONE_NAMES
from src.occlusion_estimator import compute_exposure_vector
from src.occlusion_injector import apply_occlusion, apply_all_conditions
from src.dynamic_matcher import (
    build_raw_distance_vector,
    match_dynamic_z_fusion,
    DistanceScaler,
    PATCH_PRIORS,
    HOLISTIC_PRIOR,
    MIN_ENTROPY_THRESHOLD,
    STATUS_VALID,
    STATUS_INDETERMINATE,
)
from src.evaluation import compute_far_frr, find_eer
from main import _build_registry_from_dirs


# Visual styling constants
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BG_DARK = (22, 22, 32)
_WHITE = (245, 245, 245)
_GRAY = (140, 140, 140)
_LIGHT_GRAY = (200, 200, 200)
_ACCENT_CYAN = (255, 220, 80)
_GREEN = (60, 210, 60)
_RED = (50, 50, 225)
_ORANGE = (30, 140, 255)


def render_case_image(
    probe_img_112: np.ndarray,
    gallery_img_112: np.ndarray,
    case_type: str,
    case_title: str,
    case_data: Dict[str, Any],
) -> np.ndarray:
    """
    Render a standalone 720x460 audit diagnostic card for a biometric decision case.
    Shows the probe face (with occlusion if applied), gallery face, zone exposure bars,
    dynamic weights distribution, biometric scores, operating threshold, and verdict.
    """
    canvas_w = 720
    canvas_h = 460
    canvas = np.full((canvas_h, canvas_w, 3), _BG_DARK, dtype=np.uint8)

    # Accent header bar
    header_color = _GREEN if case_type in ("TA", "TR") else _RED
    cv2.rectangle(canvas, (0, 0), (canvas_w, 4), header_color, -1)

    # Header text
    cv2.putText(canvas, f"M.A.R.K. Biometric Forensic Audit: {case_title}", (20, 32), _FONT, 0.62, _WHITE, 2, cv2.LINE_AA)
    mode_text = f"Condition: {case_data['mode']}  |  Case: {case_type} ({'CORRECT' if case_type in ('TA', 'TR') else 'WRONG'} DECISION)"
    cv2.putText(canvas, mode_text, (20, 54), _FONT, 0.40, _ACCENT_CYAN, 1, cv2.LINE_AA)

    # Panels for Probe and Gallery
    p_x, p_y = 20, 80
    g_x, g_y = 170, 80

    # Probe box
    canvas[p_y:p_y+112, p_x:p_x+112] = probe_img_112
    cv2.rectangle(canvas, (p_x-1, p_y-1), (p_x+112, p_y+112), (160, 160, 255), 1)
    cv2.putText(canvas, f"Probe: {case_data['probe_pid']}", (p_x, p_y - 8), _FONT, 0.40, (160, 160, 255), 1, cv2.LINE_AA)

    # Gallery box
    canvas[g_y:g_y+112, g_x:g_x+112] = gallery_img_112
    cv2.rectangle(canvas, (g_x-1, g_y-1), (g_x+112, g_y+112), (140, 255, 160), 1)
    cv2.putText(canvas, f"Gallery: {case_data['gallery_pid']}", (g_x, g_y - 8), _FONT, 0.40, (140, 255, 160), 1, cv2.LINE_AA)

    # Ground Truth vs Decision banner
    gt_label = case_data["label"].upper()
    pred_label = "MATCH (ACCEPTED)" if case_data["accepted"] else "REFUSED (REJECTED)"
    pred_color = _GREEN if case_type in ("TA", "TR") else _RED

    b_x, b_y = 310, 85
    cv2.putText(canvas, "Ground Truth Identity:", (b_x, b_y), _FONT, 0.42, _GRAY, 1, cv2.LINE_AA)
    gt_color = _GREEN if gt_label == "GENUINE" else _ORANGE
    cv2.putText(canvas, gt_label, (b_x + 160, b_y), _FONT, 0.45, gt_color, 2, cv2.LINE_AA)

    cv2.putText(canvas, "System Decision:", (b_x, b_y + 26), _FONT, 0.42, _GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, pred_label, (b_x + 160, b_y + 26), _FONT, 0.45, pred_color, 2, cv2.LINE_AA)

    cv2.putText(canvas, "Operating Status:", (b_x, b_y + 52), _FONT, 0.42, _GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, case_data["status"], (b_x + 160, b_y + 52), _FONT, 0.42, _WHITE, 1, cv2.LINE_AA)

    score_val = f"{case_data['score']:.4f}" if case_data['score'] is not None else "N/A"
    cv2.putText(canvas, "Fused Z-Score:", (b_x, b_y + 78), _FONT, 0.42, _GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, score_val, (b_x + 160, b_y + 78), _FONT, 0.45, _WHITE, 2, cv2.LINE_AA)

    cv2.putText(canvas, "EER Threshold:", (b_x, b_y + 104), _FONT, 0.42, _GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{case_data['threshold']:.4f}  (Match if Score <= Threshold)", (b_x + 160, b_y + 104), _FONT, 0.40, _LIGHT_GRAY, 1, cv2.LINE_AA)

    # Horizontal divider
    cv2.line(canvas, (20, 215), (canvas_w - 20, 215), (50, 50, 65), 1)

    # Zone exposures and weights breakdown
    cv2.putText(canvas, "Anatomical Zones Exposure & Active Dynamic Fusion Authority:", (20, 238), _FONT, 0.44, _ACCENT_CYAN, 1, cv2.LINE_AA)

    zones_x = 20
    zones_y = 255
    col_w = 160

    exposure = case_data["exposure"]
    raw_d = case_data["raw_5d"]
    weights = case_data["weights"]  # 5 elements [Z1, Z2, Z3, Z4, Holistic]

    for i in range(4):
        zx = zones_x + i * col_w
        e_val = exposure[i]
        bar_color = _GREEN if e_val > 0.1 else _RED

        # Zone box outline
        cv2.rectangle(canvas, (zx, zones_y), (zx + col_w - 15, zones_y + 85), (40, 40, 55), 1)
        # Zone header
        cv2.putText(canvas, f"Zone {i+1}: {ZONE_NAMES[i][:8]}", (zx + 6, zones_y + 18), _FONT, 0.38, _WHITE, 1, cv2.LINE_AA)

        # Exposure indicator bar
        cv2.rectangle(canvas, (zx + 6, zones_y + 26), (zx + col_w - 21, zones_y + 36), (30, 30, 40), -1)
        exp_fill = int((col_w - 27) * max(0.0, min(1.0, e_val)))
        if exp_fill > 0:
            cv2.rectangle(canvas, (zx + 6, zones_y + 26), (zx + 6 + exp_fill, zones_y + 36), bar_color, -1)

        cv2.putText(canvas, f"Exp: {e_val:.2f}", (zx + 6, zones_y + 52), _FONT, 0.34, _LIGHT_GRAY, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Dist: {raw_d[i]:.3f}", (zx + 6, zones_y + 66), _FONT, 0.34, _LIGHT_GRAY, 1, cv2.LINE_AA)
        w_pct = weights[i] * 100 if len(weights) > i else 0.0
        cv2.putText(canvas, f"Weight: {w_pct:.1f}%", (zx + 6, zones_y + 80), _FONT, 0.34, _ACCENT_CYAN, 1, cv2.LINE_AA)

    # Holistic information block
    hx = zones_x + 4 * col_w
    cv2.rectangle(canvas, (hx, zones_y), (canvas_w - 20, zones_y + 85), (40, 40, 55), 1)
    cv2.putText(canvas, "Global Holistic", (hx + 6, zones_y + 18), _FONT, 0.38, _WHITE, 1, cv2.LINE_AA)
    mean_exp = float(np.mean(exposure))
    cv2.putText(canvas, f"Mean Exp: {mean_exp:.2f}", (hx + 6, zones_y + 52), _FONT, 0.34, _LIGHT_GRAY, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Dist: {raw_d[4]:.3f}", (hx + 6, zones_y + 66), _FONT, 0.34, _LIGHT_GRAY, 1, cv2.LINE_AA)
    h_pct = weights[4] * 100 if len(weights) > 4 else 0.0
    cv2.putText(canvas, f"Weight: {h_pct:.1f}%", (hx + 6, zones_y + 80), _FONT, 0.34, _ACCENT_CYAN, 1, cv2.LINE_AA)

    # Diagnostic Commentary Footer
    cv2.line(canvas, (20, 360), (canvas_w - 20, 360), (50, 50, 65), 1)
    explanation = case_data.get("explanation", "")
    cv2.putText(canvas, "Forensic Diagnostic Summary:", (20, 382), _FONT, 0.40, _WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, explanation, (20, 404), _FONT, 0.36, _LIGHT_GRAY, 1, cv2.LINE_AA)

    entropy_str = f"Active Biometric Entropy: {case_data['entropy']:.4f} (Min Required Gate: {MIN_ENTROPY_THRESHOLD})"
    cv2.putText(canvas, entropy_str, (20, 430), _FONT, 0.35, _GRAY, 1, cv2.LINE_AA)

    return canvas


def generate_audit_packet(
    model_path: str = "models/face_recognizer_fast.onnx",
    gallery_dir: str = "data/gallery",
    probe_dir: str = "data/probes",
    output_dir: str = "audit_packet",
) -> None:
    """Generate the full export package."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[export] Initialising audit packet in: {output_dir}")

    # 1. Load registry & embedder
    registry = _build_registry_from_dirs(gallery_dir, probe_dir)
    embedder = FaceEmbedder(model_path=model_path)

    # 2. Pre-compute gallery embeddings
    print("[export] Pre-computing gallery embeddings...")
    gallery_data = {}
    for pid, entry in registry.items():
        img = resize_to_canonical(cv2.imread(entry["gallery_path"]))
        zones = slice_into_zones(img)
        gallery_data[pid] = {
            "img": img,
            "path": entry["gallery_path"],
            "zone_vecs": embedder.embed_zones(zones),
            "holistic_vec": embedder.embed_holistic(img),
        }

    subject_ids = sorted(gallery_data.keys())
    train_ids = subject_ids[:15]  # 60% train
    test_ids = subject_ids[15:]   # 40% test

    # 3. Fit DistanceScaler on 60% training split across all 4 conditions
    print("[export] Fitting DistanceScaler on training split...")
    raw_train_dists = []
    for pid in train_ids:
        for p_path in registry[pid]["probe_paths"]:
            p_img = resize_to_canonical(cv2.imread(p_path))
            conds = apply_all_conditions(p_img)
            for mode, occl in conds.items():
                pz = slice_into_zones(occl)
                pv = emb_zones = embedder.embed_zones(pz)
                ph = embedder.embed_holistic(occl)
                for g_pid, gdata in gallery_data.items():
                    raw_5d = build_raw_distance_vector(pv, ph, gdata["zone_vecs"], gdata["holistic_vec"])
                    raw_train_dists.append(raw_5d)

    scaler = DistanceScaler().fit(np.array(raw_train_dists))
    print(f"[export] Scaler Mean: {np.round(scaler.mean, 4)}")
    print(f"[export] Scaler Std:  {np.round(scaler.scale, 4)}")

    # 4. Evaluate test split to compute operational EER thresholds per condition
    print("[export] Scoring test split to compute exact EER thresholds...")
    cond_scores = {
        mode: {"genuine": [], "impostor": []}
        for mode in ("Baseline", "Lower_Occlusion", "Upper_Occlusion", "Dual_Occlusion")
    }
    test_comparisons = []

    for probe_pid in test_ids:
        for p_path in registry[probe_pid]["probe_paths"]:
            p_img = resize_to_canonical(cv2.imread(p_path))
            conds = apply_all_conditions(p_img)
            for mode, occl in conds.items():
                pz = slice_into_zones(occl)
                pv = embedder.embed_zones(pz)
                pe = compute_exposure_vector(pz)
                ph = embedder.embed_holistic(occl)

                for g_pid, gdata in gallery_data.items():
                    is_gen = (probe_pid == g_pid)
                    raw_5d = build_raw_distance_vector(pv, ph, gdata["zone_vecs"], gdata["holistic_vec"])
                    res = match_dynamic_z_fusion(raw_5d, pe, scaler, w_holistic=HOLISTIC_PRIOR, w_patch=PATCH_PRIORS)

                    status = res["status"]
                    score = res["distance"]

                    if status == STATUS_VALID and score is not None:
                        target_key = "genuine" if is_gen else "impostor"
                        cond_scores[mode][target_key].append(score)

                    test_comparisons.append({
                        "mode": mode,
                        "probe_pid": probe_pid,
                        "probe_path": p_path,
                        "occluded_img": occl,
                        "gallery_pid": g_pid,
                        "gallery_path": gdata["path"],
                        "gallery_img": gdata["img"],
                        "is_genuine": is_gen,
                        "status": status,
                        "score": score,
                        "entropy": res["entropy"],
                        "raw_5d": raw_5d,
                        "exposure": pe,
                        "weights": res["weights"],
                    })

    # Compute operational EER thresholds per condition
    cond_thresholds = {}
    for mode in ("Baseline", "Lower_Occlusion", "Upper_Occlusion", "Dual_Occlusion"):
        gen = cond_scores[mode]["genuine"]
        imp = cond_scores[mode]["impostor"]
        if gen and imp:
            far, frr, thresholds = compute_far_frr(gen, imp)
            eer, eer_t = find_eer(far, frr, thresholds)
            cond_thresholds[mode] = eer_t
        else:
            cond_thresholds[mode] = 0.0

    print(f"[export] Operational EER Thresholds: {cond_thresholds}")

    # 5. Classify comparisons into TA, TR, FA, FR
    candidates = {"TA": [], "TR": [], "FA": [], "FR": []}

    for item in test_comparisons:
        mode = item["mode"]
        t = cond_thresholds[mode]
        item["threshold"] = t
        status = item["status"]
        score = item["score"]
        is_gen = item["is_genuine"]

        accepted = (status == STATUS_VALID and score is not None and score <= t)
        item["accepted"] = accepted
        item["label"] = "genuine" if is_gen else "impostor"

        if is_gen and accepted:
            candidates["TA"].append(item)
        elif not is_gen and not accepted:
            candidates["TR"].append(item)
        elif not is_gen and accepted:
            candidates["FA"].append(item)
        elif is_gen and not accepted:
            candidates["FR"].append(item)

    print(f"[export] Sample pools: TA={len(candidates['TA'])}, TR={len(candidates['TR'])}, FA={len(candidates['FA'])}, FR={len(candidates['FR'])}")

    # 6. Select the 4 highly illustrative examples
    # Case 1: Correct Match (TA) - Baseline or Lower Occlusion with clear match
    sample_ta = next(c for c in candidates["TA"] if c["mode"] == "Baseline")
    sample_ta["explanation"] = f"Genuine subject {sample_ta['probe_pid']} successfully authenticated. Fused Z-score ({sample_ta['score']:.4f}) comfortably below EER threshold ({sample_ta['threshold']:.4f})."

    # Case 2: Correct Refused (TR) - Impostor under Baseline or Dual Occlusion
    sample_tr = next(c for c in candidates["TR"] if c["mode"] == "Baseline" and c["probe_pid"] != c["gallery_pid"])
    sample_tr["explanation"] = f"Impostor probe {sample_tr['probe_pid']} tested against gallery {sample_tr['gallery_pid']}. Verification correctly rejected (Score {sample_tr['score']:.4f} > Threshold {sample_tr['threshold']:.4f})."

    # Case 3: Wrongly Matching (FA) - Impostor accepted (False Accept under occlusion or disguise)
    # Prefer Upper_Occlusion or Lower_Occlusion where occlusion caused impostor distance to drop below threshold
    sample_fa = next(c for c in candidates["FA"] if c["mode"] in ("Upper_Occlusion", "Lower_Occlusion"))
    sample_fa["explanation"] = (
        f"False Accept security breach: Impostor {sample_fa['probe_pid']} incorrectly verified as gallery {sample_fa['gallery_pid']}. "
        f"{sample_fa['mode'].replace('_', ' ')} degraded facial discriminability, pulling score below threshold."
    )

    # Case 4: Wrongly Refused (FR) - Genuine rejected (False Reject under occlusion)
    sample_fr = next(c for c in candidates["FR"] if c["mode"] in ("Upper_Occlusion", "Lower_Occlusion"))
    sample_fr["explanation"] = (
        f"False Reject: Genuine subject {sample_fr['probe_pid']} incorrectly refused under {sample_fr['mode'].replace('_', ' ')}. "
        f"Occlusion elevated the fused distance to {sample_fr['score']:.4f}, exceeding the decision boundary."
    )

    cases = [
        ("TA", "Correct Match (True Accept)", sample_ta, "01_correct_matching.png"),
        ("TR", "Correct Refused (True Reject)", sample_tr, "02_correct_refused.png"),
        ("FA", "Wrongly Matching (False Accept)", sample_fa, "03_wrongly_matching.png"),
        ("FR", "Wrongly Refused (False Reject)", sample_fr, "04_wrongly_refused.png"),
    ]

    # Render and save images
    exported_samples_meta = {}
    for case_type, title, data, filename in cases:
        img_canvas = render_case_image(
            probe_img_112=data["occluded_img"],
            gallery_img_112=data["gallery_img"],
            case_type=case_type,
            case_title=title,
            case_data=data,
        )
        out_filepath = os.path.join(output_dir, filename)
        cv2.imwrite(out_filepath, img_canvas)
        print(f"[export] Saved: {out_filepath}")

        exported_samples_meta[case_type] = {
            "image_filename": filename,
            "case_title": title,
            "mode": data["mode"],
            "probe_pid": data["probe_pid"],
            "gallery_pid": data["gallery_pid"],
            "ground_truth": data["label"],
            "decision": "ACCEPTED" if data["accepted"] else "REFUSED",
            "score": float(data["score"]) if data["score"] is not None else None,
            "threshold": float(data["threshold"]),
            "status": data["status"],
            "active_entropy": float(data["entropy"]),
            "raw_distances": [float(x) for x in data["raw_5d"]],
            "exposure_scores": [float(x) for x in data["exposure"]],
            "normalized_weights": [float(x) for x in data["weights"]],
            "forensic_commentary": data["explanation"],
        }

    # 7. Generate formulas and weights documentation markdown
    formulas_md_content = f"""# M.A.R.K. Biometric Architecture: Weights & Mathematical Formulas
**Mugshot Assessment & Recognition Kernel — v4 (Dynamic Normalized Z-Fusion)**

---

## 1. System Overview
M.A.R.K. is a biometric verification engine engineered for degraded surveillance conditions (e.g., subjects wearing masks or sunglasses). Rather than relying purely on an unweighted holistic CNN, M.A.R.K. slices the canonical face ($112 \\times 112$) into four anatomical zones, assesses physical skin/texture exposure, normalizes distances to prevent scale distortion, and dynamically re-distributes decision authority.

---

## 2. Anatomical Patch Prior Weights (\\(W_{{patch}}\\) and \\(W_{{holistic}}\\))
Intrinsic biometric discriminability priors define the baseline information density of each anatomical zone:

| Region | Zone Identifier | Prior Weight (\\(W_i\\)) | Anatomical Rationale |
|---|---|---|---|
| **Forehead** | Zone 1 (\\(Z_1\\)) | **0.05** | Low individuality; often covered by hair, hats, or frontal glare. |
| **Periocular** | Zone 2 (\\(Z_2\\)) | **0.55** | **Highest authority**; captures iris, inter-ocular distance, brow bone structure. |
| **Mid-Face / Nose** | Zone 3 (\\(Z_3\\)) | **0.15** | Moderate individuality; malar structure and nasal bridge. |
| **Mouth / Jaw** | Zone 4 (\\(Z_4\\)) | **0.25** | Moderate-high individuality; lip geometry, chin contour, philtrum. |
| **Global Holistic** | Holistic (\\(H\\)) | **1.00** | Full-face baseline representation extracted by SFace CNN. |

- Base Patch Priors: `W_patch = [0.05, 0.55, 0.15, 0.25]`
- Holistic Prior: `W_holistic = 1.0` (Hybrid) or `0.0` (Patch-Only)
- Zero-Trust Entropy Gate Threshold: `MIN_ENTROPY_THRESHOLD = 0.15`

---

## 3. Mathematical Formulas

### 3.1 Cosine Distance
For any pair of L2-normalized feature embeddings \\(v_1, v_2\\):
$$\\text{{dist}}(v_1, v_2) = 1.0 - \\frac{{v_1 \\cdot v_2}}{{\\|v_1\\| \\|v_2\\|}}$$

### 3.2 Continuous Exposure Estimation (\\(E_i\\))
For each anatomical patch \\(i \\in \\{{1, 2, 3, 4\\}}\\):
1. **Skin-Tone Density (\\(S_i\\))**: Proportion of pixels satisfying human skin thresholds in the YCrCb color space:
   $$Cr \\in [133, 173], \\quad Cb \\in [77, 127]$$
2. **Texture Variance (\\(T_i\\))**: High-frequency gradient variance calculated via Sobel operator (clamped at 500.0):
   $$T_i = \\min\\left(1.0, \\frac{{\\text{{Var}}(\\nabla I)}}{{500.0}}\\right)$$
3. **Zone Exposure Score**:
   $$E_i = 0.70 \\cdot S_i + 0.30 \\cdot T_i$$
   *(Hard floor: If \\(E_i < 0.20\\), then \\(E_i \\leftarrow 0.0\\))*.
4. **Holistic Exposure Mean**:
   $$\\bar{{E}} = \\frac{{1}}{{4}} \\sum_{{i=1}}^4 E_i$$

### 3.3 Dynamic Z-Score Normalization
To prevent scale disparity between localized patch distances and global whole-face cosine distances, raw distances are transformed into dimensionless Z-scores using parameters estimated strictly on the training partition:
$$z_i = \\frac{{d_i - \\mu_i}}{{\\sigma_i}}, \\quad \\text{{for }} i \\in \\{{1, 2, 3, 4, \\text{{holistic}}\\}}$$

**Fitted Scaler Parameters (from 60% Training Split):**
- Means (\\(\\mu\\)):
  - \\(\\mu_1\\) (Forehead) = `{scaler.mean[0]:.4f}`
  - \\(\\mu_2\\) (Periocular) = `{scaler.mean[1]:.4f}`
  - \\(\\mu_3\\) (Mid-face) = `{scaler.mean[2]:.4f}`
  - \\(\\mu_4\\) (Mouth/Jaw) = `{scaler.mean[3]:.4f}`
  - \\(\\mu_{{hol}}\\) (Holistic) = `{scaler.mean[4]:.4f}`
- Scales (\\(\\sigma\\)):
  - \\(\\sigma_1\\) (Forehead) = `{scaler.scale[0]:.4f}`
  - \\(\\sigma_2\\) (Periocular) = `{scaler.scale[1]:.4f}`
  - \\(\\sigma_3\\) (Mid-face) = `{scaler.scale[2]:.4f}`
  - \\(\\sigma_4\\) (Mouth/Jaw) = `{scaler.scale[3]:.4f}`
  - \\(\\sigma_{{hol}}\\) (Holistic) = `{scaler.scale[4]:.4f}`

### 3.4 Active Authority and Dynamic Re-Normalizing Fusion
1. **Active Exposure-Weighted Authority**:
   $$\\Omega_{{\\text{{active}}}} = W_{{holistic}} \\cdot \\bar{{E}} + \\sum_{{i=1}}^4 (W_i \\cdot E_i)$$
2. **Zero-Trust Fail-Safe (Safety Gate)**:
   $$\\text{{If }} \\Omega_{{\\text{{active}}}} < 0.15 \\implies \\text{{Verdict: }} \\textbf{{INDETERMINATE\\_INSUFFICIENT\\_INFORMATION}}$$
   *(The system refuses to match rather than guessing blindly on extreme occlusions).*
3. **Dynamic Fused Distance**:
   $$Z_{{\\text{{fused}}}} = \\frac{{W_{{holistic}} \\cdot \\bar{{E}} \\cdot z_{{hol}} + \\sum_{{i=1}}^4 (W_i \\cdot E_i \\cdot z_i)}}{{\\Omega_{{\\text{{active}}}}}}$$
4. **Normalized Dynamic Fusion Weights**:
   $$w_i = \\frac{{W_i \\cdot E_i}}{{\\Omega_{{\\text{{active}}}}}}, \\quad w_{{hol}} = \\frac{{W_{{holistic}} \\cdot \\bar{{E}}}}{{\\Omega_{{\\text{{active}}}}}}$$

---

## 4. Operational Decision Rule
Given operational Equal Error Rate (EER) threshold \\(\\tau\\):
- **MATCH (ACCEPT)**: \\(Z_{{\\text{{fused}}}} \\le \\tau\\)
- **NO MATCH (REFUSE)**: \\(Z_{{\\text{{fused}}}} > \\tau\\) or \\(\\text{{Status}} = \\text{{INDETERMINATE}}\\)

**Operational Thresholds by Condition (Tri-System Benchmark):**
- Baseline: `\\tau = {cond_thresholds.get('Baseline', -1.6447):.4f}`
- Lower Occlusion (Mask): `\\tau = {cond_thresholds.get('Lower_Occlusion', -1.5088):.4f}`
- Upper Occlusion (Sunglasses): `\\tau = {cond_thresholds.get('Upper_Occlusion', -1.0769):.4f}`
- Dual Occlusion: `\\tau = {cond_thresholds.get('Dual_Occlusion', 0.2298):.4f}` *(Dual Occlusion triggers 100% FTA gate rejection)*
"""

    md_filepath = os.path.join(output_dir, "formulas_and_weights.md")
    with open(md_filepath, "w", encoding="utf-8") as f:
        f.write(formulas_md_content)
    print(f"[export] Saved: {md_filepath}")

    # 8. Save machine-readable metadata.json
    meta_json_path = os.path.join(output_dir, "metadata.json")
    complete_meta = {
        "architecture_version": "M.A.R.K. v4 (Dynamic Normalized Z-Fusion)",
        "backbone_model": "FaceRecognizerSF (SFace ONNX)",
        "embedding_dim": 128,
        "parameters": {
            "patch_priors": [float(x) for x in PATCH_PRIORS],
            "holistic_prior": float(HOLISTIC_PRIOR),
            "min_entropy_threshold": float(MIN_ENTROPY_THRESHOLD),
            "scaler_mean": [float(x) for x in scaler.mean],
            "scaler_scale": [float(x) for x in scaler.scale],
            "operational_eer_thresholds": {k: float(v) for k, v in cond_thresholds.items()},
        },
        "exported_cases": exported_samples_meta,
    }
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(complete_meta, f, indent=2)
    print(f"[export] Saved: {meta_json_path}")
    print(f"\n[export] Successfully exported all 5 items to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Export M.A.R.K. Audit Information Packet.")
    parser.add_argument("--model", default="models/face_recognizer_fast.onnx", help="Path to SFace ONNX model.")
    parser.add_argument("--gallery", default="data/gallery", help="Path to gallery folder.")
    parser.add_argument("--probes", default="data/probes", help="Path to probes folder.")
    parser.add_argument("--output", default="audit_packet", help="Output directory for audit packet.")
    args = parser.parse_args()

    generate_audit_packet(
        model_path=args.model,
        gallery_dir=args.gallery,
        probe_dir=args.probes,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
