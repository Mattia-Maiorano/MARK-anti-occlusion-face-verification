# M.A.R.K. Live Camera & Reference Photo Matcher Demonstration

`LIVE_DEMO.md` provides an operational overview of the real-time biometric matching demonstration script located at [`demo/live_camera_matcher.py`](file:///Users/Mattia/Desktop/MARK/biometric_occlusion_project/demo/live_camera_matcher.py).

---

## 1. Overview & Architecture

The live camera matcher connects a webcam feed with a static reference photo to demonstrate the **M.A.R.K. (v6 Learned Score-Level Gating & Calibrated Dual-Occlusion Recovery)** biometric architecture under real-world, unconstrained conditions.

### Architectural Pipeline
```
                          ┌──────────────────────────┐
                          │     Webcam Live Feed     │
                          └────────────┬─────────────┘
                                       │
                         [YuNet face_detection_yunet]
                                       │
              ┌────────────────────────┴────────────────────────┐
              ▼                                                 ▼
      [Face Detected]                                  [Face Missing]
              │                                                 │
   [align_face_safely]                             [Temporal Hysteresis Hold]
   (5-point / Eye-Only)                            (Preserves last 10 frames)
              │                                                 │
              └────────────────────────┬────────────────────────┘
                                       │
                            112×112 Canonical Face
                                       │
                      [spatial_slicer & FaceEmbedder]
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     ▼                                 ▼                                 ▼
1. Holistic Model             2. Patch-Only Model             3. Hybrid v6 Model
Full-face Embedding           Static Weighted Patch          Dynamic Gated Fusion &
Cosine Distance               Distance (Prior-based)          Score-Level Calibration
     │                                 │                                 │
     └─────────────────────────────────┼─────────────────────────────────┘
                                       ▼
                            Real-time HUD Interface
```

---

## 2. Real-Time HUD Metrics & Display Systems

The demonstration renders a composite 780 × 520 Heads-Up Display (HUD) presenting real-time comparative scores across **three biometric systems**:

### Systems Compared
1. **Holistic Model (Full Face)**:
   - Evaluates standard cosine distance between probe and gallery $112 \times 112$ full-face embeddings without zone breakdown.
2. **Patch-Only Model (Static)**:
   - Computes static weighted average of 4 anatomical zones using fixed discriminability priors $p = [0.20, 0.35, 0.25, 0.20]$:
     $$\text{Distance}_{\text{patch}} = \frac{\sum_{i \in \text{exposed}} p_i \cdot d(z_i^{\text{probe}}, z_i^{\text{ref}})}{\sum_{i \in \text{exposed}} p_i}$$
3. **Hybrid v6 Model (M.A.R.K. Dynamic Gated Fusion)**:
   - **Dynamic Weights**: Neural gating network predicts per-zone fusion weights $\mathbf{w}$ in real time based on exposure state $E$.
   - **Entropy Validation**: Monitors information entropy $H(\mathbf{w})$ to block decision making (`OCCLUSION BLOCKED` / `INDETERMINATE`) if facial exposure drops below security thresholds.
   - **Score Calibration**: Outputs calibrated log-odds distance scores for matching decisions.

---

## 3. Operational Features

### Robust Occlusion Alignment (`align_face_safely`)
- **Primary**: OpenCV SFace 5-landmark alignment (`recognizer.alignCrop`).
- **Fallback (`align_eyes_only`)**: Under heavy lower-face occlusion (e.g., masks, hand over mouth), YuNet's estimated mouth landmarks may become unstable. The demo automatically falls back to a similarity transformation derived strictly from the eye vector to prevent face warping.

### Temporal Hysteresis Tracker
- Holds the last valid bounding box and landmark set for up to `MAX_OCCLUSION_HOLD` (default: 10 frames) if detector confidence briefly drops due to rapid movement or sudden occlusion.

---

## 4. Key Differences: Live Demo vs. ChokePoint Benchmark Evaluation

| Dimension | ChokePoint Benchmark Suite (`main.py`) | Live Camera Demo (`demo/live_camera_matcher.py`) |
| :--- | :--- | :--- |
| **Input Source** | Pre-extracted offline frame sequences (`.jpg`). | Live USB/built-in webcam video stream (`cv2.VideoCapture`). |
| **Face Alignment** | Ground-truth XML eye coordinate annotations ($2.5 \text{ IOD} \times 3.2 \text{ IOD}$ bounding box). | Automated real-time ONNX YuNet detector (`cv2.FaceDetectorYN`) + `align_face_safely`. |
| **Enrollment Selection** | Automated selection of the single most frontal frame (`_select_best_frame`) in sequence. | User-specified static reference photo file (`--photo`). |
| **Domain Variability** | Uniform lighting, fixed focal length, and static CCTV camera angles. | Variable room lighting, pose variations, webcam lens distortion, and dynamic exposure shifts. |
| **Processing Mode** | Batch metric evaluation (EER, ROC, DET curves, rejection tables). | Real-time rate-limited loop (default 1.0s matching interval) with HUD display. |
| **Detector Threshold** | Fixed offline extraction criteria. | Configurable live score threshold (`--score-thresh`, default 0.3) + Temporal Hysteresis. |

---

## 5. Usage & Keybindings

### Execution Command
```bash
python demo/live_camera_matcher.py \
    --model models/face_recognizer_fast.onnx \
    --detector models/face_detection_yunet_2023mar.onnx \
    --photo reference.jpg \
    --camera 0 \
    [--score-thresh 0.3] \
    [--max-hold 10] \
    [--interval 1.0] \
    [--show-canonical]
```

### Interactive Controls
- `O`: Open terminal prompt to load a new reference photo path dynamically.
- `C`: Cycle webcam device index (`0`, `1`, `2`).
- `S`: Save current HUD snapshot PNG to `results/`.
- `Q` / `ESC`: Exit live matcher.
