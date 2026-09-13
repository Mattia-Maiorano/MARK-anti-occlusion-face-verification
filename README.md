# M.A.R.K. — Mugshot Assessment & Recognition Kernel

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-ONNXRuntime-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**M.A.R.K. (Mugshot Assessment & Recognition Kernel)** is an occlusion-aware biometric facial verification system engineered to maintain high authentication fidelity when subjects present with partial facial occlusions (e.g., medical masks, scarves, sunglasses, or winter clothing).

Standard deep convolutional face recognizers rely heavily on full facial geometry. When partial occlusions occur, spatial feature correlations collapse and error rates spike. M.A.R.K. addresses this vulnerability through an **Anatomical Zone Slicing** architecture coupled with a **Learned Score-Level Gating Network (v6)** and **In-Situ Live Domain Calibration (v6.1)**.

---

## Key Features

- **Anatomical Zone Slicing**: Decomposes the canonical $112 \times 112$ facial crop into 4 distinct anatomical regions ($Z_1$: Forehead, $Z_2$: Periocular, $Z_3$: Mid-Face/Nose, $Z_4$: Mouth/Jaw) alongside holistic full-face features.
- **Physical Exposure Estimation**: Computes per-zone physical exposure ($E_i \in [0, 1]$) combining skin-tone density in YCrCb color space and high-frequency Sobel texture variance.
- **Learned Score-Level Gating (v6)**: Uses an ultra-lightweight MLP operating on low-dimensional scalar distance and exposure metrics to dynamically weight facial regions based on visibility and discriminability.
- **Adaptive Rejection Gate**: Intercepts fraudulent impostor probes under severe disguises while preventing false acceptances when biometric entropy is too low.
- **Real-Time Live HUD**: Interactive webcam demonstration rendering live comparative metrics for Holistic, Patch-Only, and Calibrated Hybrid v6 models.

---

## Performance Summary (ChokePoint Benchmark)

| Operational Condition | Holistic EER | Patch-Only EER | Hybrid v6 EER | Hybrid v6 AUC | FTA Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Clean Face)** | 2.33% | 5.04% | **1.58%** | **0.9987** | 0.0% |
| **Lower Occlusion (Mask)** | 4.00% | 6.12% | **4.08%** | **0.9903** | 2.0% |
| **Upper Occlusion (Eyewear)** | 10.00% | 11.84% | **9.25%** | **0.9761** | 0.0% |
| **Dual Occlusion (Compound)** | 42.00% | 22.92% | **32.95%** | **0.7453** | 68.6% |

---

## Repository Structure

```
MARK-anti-occlusion-face-verification/
├── biometric_occlusion_project/
│   ├── src/                         # Core Python modules
│   │   ├── spatial_slicer.py        # Anatomical 4-zone facial crop & alignment
│   │   ├── occlusion_estimator.py   # YCrCb & Sobel exposure estimation
│   │   ├── embedder.py              # FaceRecognizerSF ONNX feature extraction
│   │   ├── dynamic_matcher.py       # Learned Score-Level Gating (v6) & calibration
│   │   └── evaluation.py            # Benchmark metrics, EER & ROC computation
│   ├── demo/
│   │   ├── live_camera_matcher.py   # Interactive webcam HUD verification tool
│   │   └── live_visualizer.py       # Synthetic probe & occlusion visualizer
│   ├── audit_packet/                # Forensic security audit benchmarks & charts
│   ├── results_v6/                  # EER, ROC, DET curves & metric summaries
│   ├── main.py                      # Batch benchmark runner
│   ├── calibrate_live_domain.py     # Live camera domain calibration script
│   ├── OVERVIEW.md                  # System architecture overview
│   ├── LIVE_DEMO.md                # HUD display details & live demo guide
│   ├── ACCOMPLISHMENTS_v7.md        # Comprehensive technical report
│   └── requirements.txt             # Python dependencies
└── .gitignore                       # Clean Git configuration
```

---

## Quick Start

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone git@github.com:Mattia-Maiorano/MARK-anti-occlusion-face-verification.git
cd MARK-anti-occlusion-face-verification/biometric_occlusion_project
pip install -r requirements.txt
```

### 2. Download Pretrained Models

Place the required ONNX model weights inside `biometric_occlusion_project/models/`:
- `face_recognizer_fast.onnx` (FaceRecognizerSF)
- `face_detection_yunet_2023mar.onnx` (YuNet Detector)

### 3. Run Interactive Live Webcam Matcher

Run the real-time matching demonstration against a reference mugshot photo:

```bash
python demo/live_camera_matcher.py \
  --model models/face_recognizer_fast.onnx \
  --photo path/to/gallery_reference.jpg
```

---

## Documentation

- **[Architecture Overview](biometric_occlusion_project/OVERVIEW.md)**: Deep dive into anatomical slicing, exposure estimation, and fusion mechanics.
- **[Live Demo Guide](biometric_occlusion_project/LIVE_DEMO.md)**: Guide to running the real-time webcam HUD matcher.
- **[Technical Report (v7)](biometric_occlusion_project/ACCOMPLISHMENTS_v7.md)**: Quantitative evaluation benchmarks, forensic security analysis, and v1–v6 architectural evolution.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
