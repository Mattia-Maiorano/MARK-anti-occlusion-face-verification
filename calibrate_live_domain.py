"""
calibrate_live_domain.py
------------------------
Offline calibration & MLP fine-tuning utility for Project M.A.R.K.

Ingests captured webcam calibration samples from data/calibration_samples.csv
and adapts the decision layer (DistanceScaler & LearnedGatingNetwork) to the
target webcam domain without altering the frozen SFace feature extractor.

Workflow:
  Step A: Distribution Fitting -> export models/distance_scaler_calibrated.json
  Step B: Z-Score Normalization -> construct 9D feature matrix
  Step C: Lightweight MLP Fine-Tuning -> export models/gating_v6_calibrated.pth (25 epochs)
  Step D: Verification Check -> Classification Report, EER, and Threshold Comparison
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.dynamic_matcher import (
    DistanceScaler,
    LearnedGatingNetwork,
    LearnedScoreGatingModel,
    DEFAULT_TAU_GATE,
)

MATCH_THRESHOLD = 0.40


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and optimal decision threshold for distance scores.
    Lower score = genuine (1), higher score = impostor (0).
    Returns (eer_percent, optimal_threshold).
    """
    if len(y_true) == 0:
        return 0.0, 0.40

    min_dist = float(np.min(scores))
    max_dist = float(np.max(scores))
    if abs(max_dist - min_dist) < 1e-6:
        return 0.0, float(min_dist)

    search_space = np.linspace(min_dist - 0.05, max_dist + 0.05, 500)
    
    min_diff = 1.0
    optimal_eer = 0.5
    optimal_th = 0.40

    gen_mask = (y_true == 1)
    imp_mask = (y_true == 0)

    for th in search_space:
        preds = (scores < th).astype(int)

        fnr = float(np.mean(preds[gen_mask] == 0)) if np.sum(gen_mask) > 0 else 0.0
        fpr = float(np.mean(preds[imp_mask] == 1)) if np.sum(imp_mask) > 0 else 0.0

        diff = abs(fnr - fpr)
        if diff < min_diff:
            min_diff = diff
            optimal_eer = (fnr + fpr) / 2.0
            optimal_th = float(th)

    return float(optimal_eer * 100.0), float(optimal_th)


def create_synthetic_calibration_samples(csv_path: str, n_genuine: int = 50, n_impostor: int = 50) -> None:
    """Generate representative calibration samples if CSV file does not exist."""
    print(f"[calibrate] Generating synthetic baseline calibration samples -> {csv_path}")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    np.random.seed(42)

    rows = []
    # Genuine samples (lower distance)
    for _ in range(n_genuine):
        d_z = np.random.normal(loc=[0.38, 0.35, 0.37, 0.39, 0.30], scale=[0.05, 0.05, 0.05, 0.05, 0.04])
        d_z = np.clip(d_z, 0.1, 0.6)
        E = np.random.uniform(0.7, 1.0, size=4)
        rows.append(list(d_z) + list(E) + [1.0])

    # Impostor samples (higher distance)
    for _ in range(n_impostor):
        d_z = np.random.normal(loc=[0.72, 0.68, 0.70, 0.75, 0.65], scale=[0.08, 0.08, 0.08, 0.08, 0.07])
        d_z = np.clip(d_z, 0.4, 1.2)
        E = np.random.uniform(0.7, 1.0, size=4)
        rows.append(list(d_z) + list(E) + [0.0])

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("d_z1,d_z2,d_z3,d_z4,d_holistic,E_1,E_2,E_3,E_4,label\n")
        for r in rows:
            f.write(",".join([f"{v:.6f}" for v in r]) + "\n")


def run_calibration(
    csv_path: str,
    scaler_out: str,
    weights_out: str,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> None:
    """Execute complete 4-step domain calibration utility."""
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"[calibrate] Warning: Calibration sample file '{csv_path}' not found or empty.")
        create_synthetic_calibration_samples(csv_path)

    print("\n" + "=" * 64)
    print(" M.A.R.K. In-Situ Live Domain Calibration & Fine-Tuning Pipeline")
    print("=" * 64)
    print(f" Source Sample CSV : {csv_path}")
    print(f" Target Scaler JSON: {scaler_out}")
    print(f" Target Gating PTH : {weights_out}")

    # Load dataset
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 10:
        raise ValueError(f"Expected at least 10 columns in CSV, found {data.shape[1]}")

    raw_distances = data[:, :5]  # [d_z1, d_z2, d_z3, d_z4, d_holistic]
    exposures = data[:, 5:9]     # [E_1, E_2, E_3, E_4]
    labels = data[:, 9].astype(np.float32)

    n_gen = int(np.sum(labels == 1))
    n_imp = int(np.sum(labels == 0))
    print(f" Loaded {len(data)} samples (Genuine: {n_gen}, Impostor: {n_imp})")

    # ── STEP A: Distribution Fitting ──────────────────────────────────────────
    print("\n[Step A] Fitting DistanceScaler on captured webcam distance distributions...")
    scaler = DistanceScaler()
    default_mu = scaler._DEFAULT_MEAN.copy()
    default_sigma = scaler._DEFAULT_SCALE.copy()

    scaler.fit(raw_distances)
    scaler.save_json(scaler_out)

    print(f"  • Original ChokePoint Mean μ_0 : {np.round(default_mu, 4)}")
    print(f"  • Calibrated Webcam   Mean μ   : {np.round(scaler.mean, 4)}")
    print(f"  • Original ChokePoint Std  σ_0 : {np.round(default_sigma, 4)}")
    print(f"  • Calibrated Webcam   Std  σ   : {np.round(scaler.scale, 4)}")
    print(f"  ✓ Scaler exported to {scaler_out}")

    # ── STEP B: Z-Score Normalization ─────────────────────────────────────────
    print("\n[Step B] Performing Z-Score Normalization & 9D Feature Assembly...")
    z5 = scaler.transform(raw_distances)  # (N, 5)
    X9 = np.hstack([z5, exposures]).astype(np.float32)  # (N, 9)
    print(f"  ✓ Constructed 9D feature matrix of shape {X9.shape}")

    # ── STEP C: Lightweight MLP Fine-Tuning ──────────────────────────────────
    print(f"\n[Step C] Fine-tuning 245-parameter LearnedGatingNetwork MLP ({epochs} epochs)...")
    gating_net = LearnedGatingNetwork(in_dim=9, hidden_dim=16, out_dim=5)

    v6_init_path = os.path.join(os.path.dirname(weights_out), "gating_v6.pth")
    if os.path.isfile(v6_init_path):
        try:
            gating_net.load_state_dict(torch.load(v6_init_path, map_location="cpu"))
            print(f"  • Initialized weights from pre-trained {v6_init_path}")
        except Exception as exc:
            print(f"  • Notice: Could not load pre-trained v6 weights ({exc}), using random init.")

    gating_net.train()
    optimizer = torch.optim.Adam(gating_net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    X9_t = torch.tensor(X9, dtype=torch.float32)
    E4_t = torch.tensor(exposures, dtype=torch.float32)
    Z5_t = torch.tensor(z5, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        weights_5d = gating_net(X9_t, E4_t)  # (N, 5)
        fused_z = torch.sum(weights_5d * Z5_t, dim=-1)  # (N,)
        logits = -2.0 * (fused_z - 0.0)

        loss = criterion(logits, labels_t)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {loss.item():.6f}")

    gating_net.eval()
    os.makedirs(os.path.dirname(os.path.abspath(weights_out)), exist_ok=True)
    torch.save(gating_net.state_dict(), weights_out)
    print(f"  ✓ Calibrated gating weights saved to {weights_out}")

    # ── STEP D: Verification Check ───────────────────────────────────────────
    print("\n[Step D] Verification Check & Decision Threshold Calibration...")

    raw_holistic_scores = raw_distances[:, 4]
    pre_eer, pre_thresh = compute_eer(labels, raw_holistic_scores)

    with torch.no_grad():
        final_weights = gating_net(X9_t, E4_t).cpu().numpy()
        calibrated_fused_scores = np.sum(final_weights * z5, axis=1)

    post_eer, post_thresh = compute_eer(labels, calibrated_fused_scores)

    binary_preds = (calibrated_fused_scores < post_thresh).astype(int)
    acc = accuracy_score(labels, binary_preds)
    prec = precision_score(labels, binary_preds, zero_division=0)
    rec = recall_score(labels, binary_preds, zero_division=0)
    f1 = f1_score(labels, binary_preds, zero_division=0)

    print("\n" + "─" * 50)
    print(" CALIBRATION VERIFICATION SUMMARY REPORT")
    print("─" * 50)
    print(f" Pre-Calibration  (Holistic Raw) EER : {pre_eer:.2f}%  | Thresh: {pre_thresh:.4f}")
    print(f" Post-Calibration (Hybrid Gated) EER : {post_eer:.2f}%  | Thresh: {post_thresh:.4f}")
    print(f" Accuracy : {acc * 100:.2f}%")
    print(f" Precision: {prec:.4f}")
    print(f" Recall   : {rec:.4f}")
    print(f" F1-Score : {f1:.4f}")
    print("─" * 50)
    print("\nDetailed Classification Report:")
    print(classification_report(labels, binary_preds, target_names=["Impostor (0)", "Genuine (1)"], zero_division=0))
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="M.A.R.K. In-Situ Live Domain Calibration & Fine-Tuning Utility",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--csv", default="data/calibration_samples.csv", help="Path to input calibration samples CSV file.")
    parser.add_argument("--scaler-out", default="models/distance_scaler_calibrated.json", help="Path to output calibrated DistanceScaler JSON file.")
    parser.add_argument("--weights-out", default="models/gating_v6_calibrated.pth", help="Path to output calibrated LearnedGatingNetwork weights.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of fine-tuning epochs for LearnedGatingNetwork (default: 25).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer (default: 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for Adam optimizer (default: 1e-4).")

    args = parser.parse_args()
    run_calibration(
        csv_path=args.csv,
        scaler_out=args.scaler_out,
        weights_out=args.weights_out,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


if __name__ == "__main__":
    main()
