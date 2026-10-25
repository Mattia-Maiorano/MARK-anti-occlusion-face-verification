"""
evaluation.py
-------------
Biometric Statistical Evaluation Suite — M.A.R.K. v6 (Learned Score-Level Gating & Calibrated Rejection).

Comparative benchmark across three biometric verification systems:
    • Holistic Baseline    — Standard whole-face SFace cosine distance (unchanged)
    • M.A.R.K. Patch-Only — Learned Gating Network operating on patch-only subspace
    • M.A.R.K. Hybrid     — Learned Gating Network (9D vector, continuous exposure soft-masking)

Two-Pass Evaluation Protocol
-----------------------------
  Pass 1 (Training — multi-sequence corpus):
    1. Extract face crops and compute embeddings for training sequences.
    2. Fit PatchDomainAdapters on visible patches to mitigate crop domain gap.
    3. Fit DistanceScaler exclusively on training split distance vectors across all conditions.
    4. Train LearnedGatingNetwork (< 500 params) on genuine and impostor 9D distance-exposure pairs.
    5. Calibrate adaptive confidence gate threshold tau_gate via ROC/DET on training split.

  Pass 2 (Test — fixed evaluation subset):
    Score all test pairs on the 40% held-out subject partition (P1E_S1_C1):
      - Holistic Baseline: raw cosine distance d_hol
      - Patch-Only v6:     gating network with holistic authority zeroed
      - Hybrid v6:         full 9D learned gating network
    Track INDETERMINATE rejections and compute forensic classifications.

Third-party : NumPy, SciPy, Matplotlib, PyTorch
Custom      : Two-pass runner, tri-system metrics, rejection forensics, publication plotting.
"""

from __future__ import annotations

import csv
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from .spatial_slicer import resize_to_canonical, slice_into_zones
from .occlusion_injector import apply_all_conditions, OCCLUSION_MODES
from .occlusion_estimator import compute_exposure_vector
from .dynamic_matcher import (
    LearnedScoreGatingModel,
    DistanceScaler,
    cosine_distance,
    build_raw_distance_vector,
    PATCH_PRIORS,
    HOLISTIC_PRIOR,
    MIN_ENTROPY_THRESHOLD,
    STATUS_VALID,
    STATUS_INDETERMINATE,
)

# ── Benchmark Parameters ──────────────────────────────────────────────────────
_N_THRESHOLD_STEPS = 2000
_TRAIN_SPLIT_RATIO = 0.60

# ── Visual Styling Constants ──────────────────────────────────────────────────
_COND_COLORS = {
    "Baseline":        "#4CAF50",
    "Lower_Occlusion": "#2196F3",
    "Upper_Occlusion": "#FF9800",
    "Dual_Occlusion":  "#EF5350",
}
_HOLISTIC_KW = {"linewidth": 1.5, "linestyle": "--", "alpha": 0.80}
_PATCH_KW    = {"linewidth": 1.8, "linestyle": ":",  "alpha": 0.85}
_HYBRID_KW   = {"linewidth": 2.4, "linestyle": "-",  "alpha": 0.95}
_MARK_KW     = _PATCH_KW


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — IMAGE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_resize(image_path: str) -> Optional[np.ndarray]:
    """Load BGR image from disk and resize to canonical 112 × 112."""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] Cannot read image: '{image_path}'")
        return None
    try:
        return resize_to_canonical(img)
    except ValueError as exc:
        print(f"  [WARN] Resize failed for '{image_path}': {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT (Subject-Level Partitioning)
# ─────────────────────────────────────────────────────────────────────────────

def _split_subjects(
    subject_ids: List[str], ratio: float = _TRAIN_SPLIT_RATIO
) -> Tuple[List[str], List[str]]:
    """Subject-level deterministic partition preventing cross-identity data leakage."""
    n_train   = max(1, int(len(subject_ids) * ratio))
    train_ids = subject_ids[:n_train]
    test_ids  = subject_ids[n_train:]
    return train_ids, test_ids


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SEQUENCE TRAINING CORPUS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _discover_training_sequences(
    training_data_dirs: List[str],
    groundtruth_dir:    str,
) -> List[Tuple[str, str]]:
    """
    Auto-discover valid (sequence_dir, xml_path) pairs for multi-sequence training.
    """
    pairs: List[Tuple[str, str]] = []

    if not os.path.isdir(groundtruth_dir):
        print(f"[training] Warning: groundtruth_dir not found: '{groundtruth_dir}'")
        return pairs

    for parent_dir in training_data_dirs:
        if not os.path.isdir(parent_dir):
            print(f"[training] Warning: training_data_dir not found: '{parent_dir}'")
            continue
        for seq_name in sorted(os.listdir(parent_dir)):
            seq_dir  = os.path.join(parent_dir, seq_name)
            if not os.path.isdir(seq_dir):
                continue
            xml_path = os.path.join(groundtruth_dir, f"{seq_name}.xml")
            if os.path.isfile(xml_path):
                pairs.append((seq_dir, xml_path))
                print(f"  [training] Discovered: {seq_name} + {os.path.basename(xml_path)}")
            else:
                print(f"  [training] No XML for '{seq_name}' in groundtruth — skipping.")

    return pairs


def _build_extra_training_samples(
    training_sequences: List[Tuple[str, str]],
    embedder,
    results_dir: str,
) -> List[Dict]:
    """
    Extract probe embeddings from extra ChokePoint training sequences (multi-camera C2, C3).
    """
    from .chokepoint_extractor import extract_chokepoint_dataset

    tmp_root = os.path.join(results_dir, "training_tmp")
    os.makedirs(tmp_root, exist_ok=True)

    extra_samples: List[Dict] = []

    for seq_dir, xml_path in training_sequences:
        seq_name = os.path.basename(seq_dir)
        gal_dir  = os.path.join(tmp_root, seq_name, "gallery")
        prb_dir  = os.path.join(tmp_root, seq_name, "probes")
        os.makedirs(gal_dir, exist_ok=True)
        os.makedirs(prb_dir, exist_ok=True)

        try:
            registry = extract_chokepoint_dataset(
                seq_dir, xml_path, gal_dir, prb_dir
            )
        except Exception as exc:
            print(f"  [training] Extraction failed for '{seq_name}': {exc}")
            continue

        n_before = len(extra_samples)

        for pid, entry in sorted(registry.items()):
            for probe_path in entry.get("probe_paths", []):
                raw_112 = _load_and_resize(probe_path)
                if raw_112 is None:
                    continue

                all_conditions = apply_all_conditions(raw_112)
                for mode, occluded in all_conditions.items():
                    p_zones = slice_into_zones(occluded)
                    z_vecs  = embedder.embed_zones(p_zones)
                    exp_vec = compute_exposure_vector(p_zones)
                    h_vec   = embedder.embed_holistic(occluded)

                    extra_samples.append({
                        "pid":          pid,
                        "mode":         mode,
                        "zone_vecs":    np.array(z_vecs,   dtype=np.float32),
                        "holistic_vec": np.array(h_vec,    dtype=np.float32),
                        "exposure":     np.array(exp_vec,  dtype=np.float32),
                    })

        n_added = len(extra_samples) - n_before
        print(f"  [training] {seq_name}: added {n_added} samples.")

    return extra_samples


# ─────────────────────────────────────────────────────────────────────────────
# TWO-PASS TRI-SYSTEM EVALUATION RUNNER (M.A.R.K. v6)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    registry:            Dict,
    embedder,
    results_dir:         str       = "results",
    training_data_dirs:  Optional[List[str]] = None,
    groundtruth_dir:     Optional[str]       = None,
) -> Dict:
    """
    Execute the full M.A.R.K. v6 biometric evaluation suite:
      Pass 1: Train PatchDomainAdapters on multi-camera corpus, fit DistanceScaler,
              train LearnedGatingNetwork, and calibrate adaptive gate on training split.
      Pass 2: Tri-system Learned Score-Level Gating scoring on test split.
    """
    os.makedirs(results_dir, exist_ok=True)

    # ── Pre-compute gallery embeddings ───────────────────────────────────────
    print("\n[evaluation] Computing gallery embeddings...")
    gallery_data: Dict[str, Dict] = {}

    for pid, entry in registry.items():
        img_112 = _load_and_resize(entry["gallery_path"])
        if img_112 is None:
            continue

        zones        = slice_into_zones(img_112)
        zone_vecs    = embedder.embed_zones(zones)
        holistic_vec = embedder.embed_holistic(img_112)

        gallery_data[pid] = {
            "zone_vecs":    np.array(zone_vecs,    dtype=np.float32),
            "holistic_vec": np.array(holistic_vec, dtype=np.float32),
            "exposure":     np.ones(4, dtype=np.float32),
        }
        print(f"  ✓ '{pid}' gallery enrolled.")

    if len(gallery_data) < 2:
        raise RuntimeError("[evaluation] At least 2 subjects required in gallery.")

    subject_ids         = sorted(gallery_data.keys())
    train_ids, test_ids = _split_subjects(subject_ids)
    print(
        f"\n[evaluation] Gallery ready: {len(subject_ids)} subjects.\n"
        f"[evaluation] Train/Test partition: {len(train_ids)} train / {len(test_ids)} test."
    )

    # ═════════════════════════════════════════════════════════════════════════
    # PASS 1 — Training: PatchDomainAdapters + DistanceScaler + LearnedGatingNet
    # ═════════════════════════════════════════════════════════════════════════
    print(
        "\n[evaluation] ── PASS 1: Training M.A.R.K. v6 Learned Score-Level Gating Model ──"
    )

    fusion_model = LearnedScoreGatingModel()

    # ── Baseline training samples (P1E_S1_C1, train split only) ─────────────
    print(f"  [training] Collecting baseline probe samples from train split ({len(train_ids)} subjects)...")
    train_samples: List[Dict] = []
    for probe_pid in train_ids:
        for probe_path in registry[probe_pid].get("probe_paths", []):
            raw_112 = _load_and_resize(probe_path)
            if raw_112 is None:
                continue

            all_conditions = apply_all_conditions(raw_112)
            for mode, occluded in all_conditions.items():
                p_zones = slice_into_zones(occluded)
                z_vecs  = embedder.embed_zones(p_zones)
                exp_vec = compute_exposure_vector(p_zones)
                h_vec   = embedder.embed_holistic(occluded)

                train_samples.append({
                    "pid":          probe_pid,
                    "mode":         mode,
                    "zone_vecs":    np.array(z_vecs,  dtype=np.float32),
                    "holistic_vec": np.array(h_vec,   dtype=np.float32),
                    "exposure":     np.array(exp_vec, dtype=np.float32),
                })

    print(f"  [training] Baseline corpus: {len(train_samples)} samples "
          f"({len(train_ids)} subjects × probes × 4 conditions).")

    # ── Extra training samples (multi-sequence expansion) ────────────────────
    extra_samples: List[Dict] = []
    if training_data_dirs and groundtruth_dir:
        print("\n  [training] Discovering additional training sequences...")
        training_seqs = _discover_training_sequences(
            training_data_dirs, groundtruth_dir
        )
        # Exclude main evaluation sequence from extra sequences to prevent double-counting
        c2_c3_seqs = [p for p in training_seqs if not p[0].endswith("P1E_S1_C1")]
        if c2_c3_seqs:
            print(f"\n  [training] Building multi-sequence corpus from {len(c2_c3_seqs)} sequences...")
            extra_samples = _build_extra_training_samples(
                c2_c3_seqs, embedder, results_dir
            )
        else:
            print("  [training] No additional sequences discovered.")
    else:
        print("  [training] No extra training dirs provided — using baseline corpus only.")

    # ── Multi-camera training split (restricted strictly to train_ids for zero leakage) ──
    filtered_extra = [s for s in extra_samples if s["pid"] in train_ids]
    all_train_samples = train_samples + filtered_extra
    print(
        f"\n  [training] Total training corpus for domain adapters: {len(all_train_samples)} samples "
        f"(baseline: {len(train_samples)}, extra multi-camera: {len(filtered_extra)})."
    )

    # ── Fit PatchDomainAdapters ───────────────────────────────────────────────
    fusion_model.fit_adapters(
        train_samples=all_train_samples,
        gallery_data=gallery_data,
        train_ids=train_ids,
        epochs=30,
        lr=1e-3,
    )

    # ── Fit DistanceScaler exclusively on training split ─────────────────────
    fusion_model.fit_scaler(
        train_samples=train_samples,
        gallery_data=gallery_data,
    )

    # ── Train LearnedGatingNetwork (< 500 parameters) ────────────────────────
    fusion_model.fit_gating_network(
        train_samples=train_samples,
        gallery_data=gallery_data,
        epochs=200,
        lr=8e-3,
    )

    # ── Calibrate Adaptive Gate ──────────────────────────────────────────────
    fusion_model.calibrate_adaptive_gate(
        train_samples=train_samples,
    )

    # ── Load Live Domain Calibrated Parameters if present ─────────────────────
    calib_scaler_path = "models/distance_scaler_calibrated.json"
    calib_gating_path = "models/gating_v6_calibrated.pth"
    if os.path.isfile(calib_scaler_path):
        print(f"  [evaluation] Overriding DistanceScaler with live domain calibrated scaler: {calib_scaler_path}")
        fusion_model.load_calibrated_scaler(calib_scaler_path)
    if os.path.isfile(calib_gating_path):
        print(f"  [evaluation] Overriding LearnedGatingNetwork with live domain calibrated weights: {calib_gating_path}")
        fusion_model.load_calibrated_gating_weights(calib_gating_path)

    # ═════════════════════════════════════════════════════════════════════════
    # PASS 2 — Test split: tri-system Learned Score-Level Gating scoring
    # ═════════════════════════════════════════════════════════════════════════
    print(
        "\n[evaluation] ── PASS 2: Scoring test split (M.A.R.K. v6 Learned Score-Level Gating) ──"
    )

    def _empty_cond():
        return {
            "mark":                {"genuine": [], "impostor": []},
            "holistic":            {"genuine": [], "impostor": []},
            "hybrid":              {"genuine": [], "impostor": []},
            "indeterminate_count": 0,
            "rejected_pairs":      [],
        }

    results: Dict[str, Dict] = {mode: _empty_cond() for mode in OCCLUSION_MODES}

    for probe_pid in test_ids:
        for probe_path in registry[probe_pid].get("probe_paths", []):
            raw_112 = _load_and_resize(probe_path)
            if raw_112 is None:
                continue

            all_conditions = apply_all_conditions(raw_112)

            for mode, occluded in all_conditions.items():
                p_zones = slice_into_zones(occluded)
                p_zvecs = embedder.embed_zones(p_zones)
                p_exp   = compute_exposure_vector(p_zones)
                p_hvec  = embedder.embed_holistic(occluded)

                for gallery_pid, gdata in gallery_data.items():
                    key = "genuine" if (probe_pid == gallery_pid) else "impostor"

                    # ── 1. Holistic Baseline (raw cosine distance) ─────────
                    h_dist = cosine_distance(p_hvec, gdata["holistic_vec"])
                    results[mode]["holistic"][key].append(h_dist)

                    # ── 2. M.A.R.K. v6 Patch-Only Gating ───────────────────
                    patch_res = fusion_model.match(
                        probe_zones=p_zvecs,
                        probe_holistic=p_hvec,
                        probe_exposure=p_exp,
                        gallery_zones=gdata["zone_vecs"],
                        gallery_holistic=gdata["holistic_vec"],
                        mode="patch_only",
                    )

                    if patch_res["status"] != STATUS_VALID:
                        results[mode]["indeterminate_count"] += 1
                        results[mode]["rejected_pairs"].append({
                            "probe_pid":     probe_pid,
                            "gallery_pid":   gallery_pid,
                            "label":         key,
                            "holistic_dist": h_dist,
                        })
                    else:
                        results[mode]["mark"][key].append(patch_res["distance"])

                    # ── 3. M.A.R.K. v6 Hybrid Learned Gating ───────────────
                    hybrid_res = fusion_model.match(
                        probe_zones=p_zvecs,
                        probe_holistic=p_hvec,
                        probe_exposure=p_exp,
                        gallery_zones=gdata["zone_vecs"],
                        gallery_holistic=gdata["holistic_vec"],
                        mode="hybrid",
                    )

                    if hybrid_res["status"] == STATUS_VALID:
                        results[mode]["hybrid"][key].append(hybrid_res["distance"])

    # Score collection summary
    print("\n[evaluation] Score collection complete:")
    for mode in OCCLUSION_MODES:
        r = results[mode]
        total = (
            len(r["hybrid"]["genuine"]) + len(r["hybrid"]["impostor"])
            + r["indeterminate_count"]
        )
        rr = r["indeterminate_count"] / total * 100 if total > 0 else 0
        print(
            f"  {mode:<20}  genuine={len(r['hybrid']['genuine']):<5}  "
            f"impostor={len(r['hybrid']['impostor']):<6}  "
            f"indet={r['indeterminate_count']}  ({rr:.1f}%)"
        )

    results["fusion_model"] = fusion_model
    return results


# ─────────────────────────────────────────────────────────────────────────────
# REJECTION FORENSICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rejection_forensics(raw_results: Dict, metrics: Dict) -> Dict:
    """
    Classify each INDETERMINATE M.A.R.K. rejection by what the holistic
    baseline would have decided at its EER operating threshold:
      TA (True Accept)  — genuine pair; holistic would have accepted
      FA (False Accept) — impostor pair; holistic would have accepted (security breach prevented)
      TR (True Reject)  — impostor pair; holistic would have rejected
      FR (False Reject) — genuine pair; holistic would have rejected
    """
    forensics = {}

    for mode in OCCLUSION_MODES:
        r          = raw_results[mode]
        rejected   = r.get("rejected_pairs", [])
        m_hol      = metrics[mode]["holistic"]
        eer_thresh = m_hol.get("eer_threshold", 0.5)

        total_comps = (
            len(r["hybrid"]["genuine"]) + len(r["hybrid"]["impostor"])
            + r["indeterminate_count"]
        )
        rej_rate = (
            r["indeterminate_count"] / total_comps * 100 if total_comps > 0 else 0.0
        )

        rows   = []
        counts = {"TA": 0, "FA": 0, "TR": 0, "FR": 0}

        for pair in rejected:
            h_dist   = pair["holistic_dist"]
            label    = pair["label"]
            h_accept = h_dist <= eer_thresh

            if label == "genuine":
                cls = "TA" if h_accept else "FR"
            else:
                cls = "FA" if h_accept else "TR"

            counts[cls] += 1
            rows.append({
                "probe_pid":         pair["probe_pid"],
                "gallery_pid":       pair["gallery_pid"],
                "label":             label,
                "holistic_dist":     round(h_dist, 6),
                "eer_threshold":     round(eer_thresh, 6),
                "holistic_decision": "ACCEPT" if h_accept else "REJECT",
                "forensic_class":    cls,
            })

        forensics[mode] = {
            "total_rejected": r["indeterminate_count"],
            "rejection_rate": rej_rate,
            "TA":             counts["TA"],
            "FA":             counts["FA"],
            "TR":             counts["TR"],
            "FR":             counts["FR"],
            "fa_prevented":   counts["FA"],
            "rows":           rows,
        }

    return forensics


def save_rejection_forensics_csv(forensics: Dict, save_path: str) -> None:
    """Save per-pair forensic rejection classifications to CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    all_rows = []
    fieldnames = [
        "condition", "probe_pid", "gallery_pid", "label",
        "holistic_dist", "eer_threshold", "holistic_decision", "forensic_class",
    ]
    for mode in OCCLUSION_MODES:
        for row in forensics[mode]["rows"]:
            all_rows.append({"condition": mode, **row})

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[evaluation] Rejection forensics CSV → {save_path}")


def print_rejection_forensics_table(forensics: Dict) -> None:
    """Print formatted rejection forensics summary table to stdout."""
    sep = "─" * 88
    print()
    print(sep)
    print("  M.A.R.K. v6 Adaptive Gate — Rejection Forensic Analysis")
    print(sep)
    print(
        f"  {'Condition':<20}  {'Total Rej':>9}  {'FTA %':>6}  "
        f"{'TA':>5}  {'FA (prev.)':>10}  {'TR':>5}  {'FR':>5}"
    )
    print(sep)

    total_fa = 0
    for mode in OCCLUSION_MODES:
        f = forensics[mode]
        total_fa += f["fa_prevented"]
        print(
            f"  {mode:<20}  {f['total_rejected']:>9}  "
            f"{f['rejection_rate']:>5.1f}%  "
            f"{f['TA']:>5}  {f['fa_prevented']:>10}  "
            f"{f['TR']:>5}  {f['FR']:>5}"
        )

    print(sep)
    print(f"\n  ✓ Adaptive gate prevented {total_fa} Impostor False Accepts while maintaining operational utility.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# METRIC COMPUTATION (FAR, FRR, EER, AUC)
# ─────────────────────────────────────────────────────────────────────────────

def compute_far_frr(
    genuine_scores:  List[float],
    impostor_scores: List[float],
    n_steps:         int = _N_THRESHOLD_STEPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep distance thresholds to calculate FAR and FRR curves."""
    gen = np.array(genuine_scores,  dtype=np.float64)
    imp = np.array(impostor_scores, dtype=np.float64)

    all_scores = np.concatenate([gen, imp])
    s_min = float(np.min(all_scores))
    s_max = float(np.max(all_scores))
    span  = max(1e-4, s_max - s_min)
    t_min = s_min - 0.05 * span
    t_max = s_max + 0.05 * span
    thresholds = np.linspace(t_min, t_max, n_steps)

    far = np.array([np.mean(imp <= t) for t in thresholds], dtype=np.float64)
    frr = np.array([np.mean(gen  > t) for t in thresholds], dtype=np.float64)
    return far, frr, thresholds


def find_eer(
    far:        np.ndarray,
    frr:        np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float]:
    """Locate Equal Error Rate (EER) and crossing threshold via linear interpolation."""
    diff         = far - frr
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        idx = int(np.argmin(np.abs(diff)))
        return float((far[idx] + frr[idx]) / 2.0), float(thresholds[idx])

    idx = sign_changes[0]
    t0, t1 = float(thresholds[idx]), float(thresholds[idx + 1])
    d0, d1 = float(diff[idx]),       float(diff[idx + 1])

    denom = d1 - d0
    t_eer = t0 - d0 * (t1 - t0) / denom if abs(denom) > 1e-12 else (t0 + t1) / 2.0
    t_eer = float(np.clip(t_eer, thresholds[0], thresholds[-1]))

    f_far = interpolate.interp1d(thresholds, far, kind="linear")
    f_frr = interpolate.interp1d(thresholds, frr, kind="linear")
    eer   = float((f_far(t_eer) + f_frr(t_eer)) / 2.0)
    return eer, t_eer


def compute_auc(far: np.ndarray, tar: np.ndarray) -> float:
    """Compute Area Under ROC Curve (AUC) via trapezoidal integration."""
    idx   = np.argsort(far)
    far_s = far[idx]
    tar_s = tar[idx]
    dx    = np.diff(far_s)
    avg_y = (tar_s[:-1] + tar_s[1:]) / 2.0
    return float(np.sum(dx * avg_y))


def _dummy_metrics() -> Dict:
    t = np.linspace(0, 2, _N_THRESHOLD_STEPS)
    return {
        "far":           np.linspace(0, 1, _N_THRESHOLD_STEPS),
        "frr":           np.linspace(1, 0, _N_THRESHOLD_STEPS),
        "tar":           np.linspace(0, 1, _N_THRESHOLD_STEPS),
        "thresholds":    t,
        "eer":           0.5,
        "eer_threshold": 1.0,
        "auc":           0.5,
    }


def compute_all_metrics(results: Dict) -> Dict:
    """Assemble FAR, FRR, TAR, EER, AUC across all conditions × 3 systems."""
    metrics: Dict[str, Dict] = {}

    for mode in OCCLUSION_MODES:
        r = results[mode]
        metrics[mode] = {}

        for system in ("mark", "holistic", "hybrid"):
            gen = r[system]["genuine"]
            imp = r[system]["impostor"]

            if not gen or not imp:
                metrics[mode][system] = _dummy_metrics()
                continue

            far, frr, thresh = compute_far_frr(gen, imp)
            eer, eer_t       = find_eer(far, frr, thresh)
            tar              = 1.0 - frr
            auc              = compute_auc(far, tar)

            metrics[mode][system] = {
                "far":           far,
                "frr":           frr,
                "tar":           tar,
                "thresholds":    thresh,
                "eer":           eer,
                "eer_threshold": eer_t,
                "auc":           auc,
            }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# PUBLICATION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_distributions(results: Dict, save_path: str) -> None:
    """Genuine vs Impostor distance distributions (M.A.R.K. v6 Learned Score-Level Gating)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "M.A.R.K. v6  —  Genuine vs Impostor Score Distributions"
        "  (Learned Score-Level Gating)",
        fontsize=15, fontweight="bold",
    )

    for ax, mode in zip(axes.flat, OCCLUSION_MODES):
        gen   = results[mode]["hybrid"]["genuine"]
        imp   = results[mode]["hybrid"]["impostor"]
        color = _COND_COLORS[mode]

        if gen:
            ax.hist(gen, bins=50, density=True, alpha=0.72,
                    color="#43A047", label=f"Genuine (n={len(gen)})")
        if imp:
            ax.hist(imp, bins=50, density=True, alpha=0.60,
                    color="#E53935", label=f"Impostor (n={len(imp)})")

        ax.set_title(f"{mode.replace('_', ' ')}", color=color, fontweight="bold")
        ax.set_xlabel("Fused Z-Score Distance")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluation] Score distributions → {save_path}")


def plot_roc_curves(results: Dict, metrics: Dict, save_path: str) -> None:
    """Tri-System ROC Curves (Holistic vs Patch-Only v6 vs Hybrid v6)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "M.A.R.K. v6  —  Tri-System ROC Curves (Learned Score-Level Gating)",
        fontsize=15, fontweight="bold",
    )

    for ax, mode in zip(axes.flat, OCCLUSION_MODES):
        color = _COND_COLORS[mode]
        m     = metrics[mode]

        ax.plot(
            m["holistic"]["far"], m["holistic"]["tar"],
            color="#757575",
            label=f"Holistic Baseline  AUC={m['holistic']['auc']:.4f}  EER={m['holistic']['eer']*100:.1f}%",
            **_HOLISTIC_KW,
        )
        ax.plot(
            m["mark"]["far"], m["mark"]["tar"],
            color=color,
            label=f"v6 Patch-Only      AUC={m['mark']['auc']:.4f}  EER={m['mark']['eer']*100:.1f}%",
            **_PATCH_KW,
        )
        ax.plot(
            m["hybrid"]["far"], m["hybrid"]["tar"],
            color=color,
            label=f"v6 Hybrid          AUC={m['hybrid']['auc']:.4f}  EER={m['hybrid']['eer']*100:.1f}%",
            **_HYBRID_KW,
        )

        ax.plot([0, 1], [0, 1], "k:", alpha=0.30, label="Chance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(mode.replace("_", " "), color=color, fontweight="bold")
        ax.set_xlabel("False Accept Rate (FAR)")
        ax.set_ylabel("True Accept Rate (TAR = 1 - FRR)")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluation] ROC curves → {save_path}")


def plot_det_curves(results: Dict, metrics: Dict, save_path: str) -> None:
    """Tri-System DET Curves (Log-scale FAR vs FNR)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "M.A.R.K. v6  —  Tri-System DET Curves"
        "  (Holistic / Patch-Only v6 / Hybrid v6)",
        fontsize=15, fontweight="bold",
    )

    for ax, mode in zip(axes.flat, OCCLUSION_MODES):
        color  = _COND_COLORS[mode]
        m      = metrics[mode]

        h_fnr  = np.clip(1.0 - m["holistic"]["tar"], 1e-4, 1.0)
        p_fnr  = np.clip(1.0 - m["mark"]["tar"],     1e-4, 1.0)
        hy_fnr = np.clip(1.0 - m["hybrid"]["tar"],   1e-4, 1.0)
        far_h  = np.clip(m["holistic"]["far"],        1e-4, 1.0)
        far_p  = np.clip(m["mark"]["far"],            1e-4, 1.0)
        far_hy = np.clip(m["hybrid"]["far"],          1e-4, 1.0)

        ax.plot(far_h,  h_fnr,  color="#757575",
                label=f"Holistic Baseline  EER={m['holistic']['eer']*100:.1f}%", **_HOLISTIC_KW)
        ax.plot(far_p,  p_fnr,  color=color,
                label=f"v6 Patch-Only      EER={m['mark']['eer']*100:.1f}%",     **_PATCH_KW)
        ax.plot(far_hy, hy_fnr, color=color,
                label=f"v6 Hybrid          EER={m['hybrid']['eer']*100:.1f}%",   **_HYBRID_KW)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 1.0)
        ax.set_ylim(1e-3, 1.0)
        ax.set_title(mode.replace("_", " "), color=color, fontweight="bold")
        ax.set_xlabel("False Accept Rate (FAR)")
        ax.set_ylabel("False Non-Match Rate (FNR = FRR)")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.25, which="both")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluation] DET curves → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# TABLES & CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_csv(metrics: Dict, save_path: str) -> None:
    """Save 6-column EER and AUC comparative table to CSV."""
    rows = []
    for mode in OCCLUSION_MODES:
        m = metrics[mode]
        rows.append({
            "Condition":       mode,
            "Holistic_EER_%":  f"{m['holistic']['eer'] * 100:.2f}",
            "PatchOnly_EER_%": f"{m['mark']['eer']     * 100:.2f}",
            "Hybrid_EER_%":    f"{m['hybrid']['eer']   * 100:.2f}",
            "Holistic_AUC":    f"{m['holistic']['auc']:.4f}",
            "PatchOnly_AUC":   f"{m['mark']['auc']:.4f}",
            "Hybrid_AUC":      f"{m['hybrid']['auc']:.4f}",
        })

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[evaluation] Metrics CSV → {save_path}")


def print_metrics_table(metrics: Dict) -> None:
    """Print formatted 6-column EER / AUC summary table to stdout."""
    header = (
        f"{'Condition':<22} "
        f"{'Holistic EER':>13} "
        f"{'Patch EER':>12} "
        f"{'Hybrid EER':>12} "
        f"{'Holistic AUC':>13} "
        f"{'Patch AUC':>10} "
        f"{'Hybrid AUC':>11}"
    )
    sep = "─" * len(header)

    print()
    print(sep)
    print("  M.A.R.K. v6 — Learned Score-Level Gating & Calibrated Rejection")
    print(sep)
    print(header)
    print(sep)

    for mode in OCCLUSION_MODES:
        m = metrics[mode]
        print(
            f"{mode:<22} "
            f"{m['holistic']['eer'] * 100:>12.2f}% "
            f"{m['mark']['eer']     * 100:>11.2f}% "
            f"{m['hybrid']['eer']   * 100:>11.2f}% "
            f"{m['holistic']['auc']:>13.4f} "
            f"{m['mark']['auc']:>10.4f} "
            f"{m['hybrid']['auc']:>11.4f}"
        )

    print(sep)
    print("  Architecture: 4× PatchDomainAdapter + DistanceScaler + LearnedGatingNetwork (<500 params)")
    print(sep)
    print()
