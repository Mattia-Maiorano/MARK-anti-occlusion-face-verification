Project Overview: M.A.R.K. (Mugshot Assessment & Recognition Kernel) — v5 Architecture

Goal
M.A.R.K. is engineered to address a critical vulnerability in modern surveillance: the severe degradation of facial recognition accuracy when subjects present with partial facial occlusions. The primary objective is to build a robust biometric verification system capable of securely matching pristine, enrolled mugshots against uncooperative, degraded CCTV probes—such as individuals wearing medical masks, sunglasses, or heavy winter clothing. By shifting away from rigid late-stage score fusion to adaptive early feature-level attention fusion, the system preserves non-linear geometric correlations across visible regions and maintains high-confidence authentication even when significant portions of the identity matrix are occluded.

Architecture
The system utilizes a hybrid architecture that pairs a frozen, off-the-shelf Deep Convolutional Neural Network (FaceRecognizerSF) with a physics-informed feature-level attention engine:
1. Anatomical Zone Slicing: The canonical $112 \times 112$ facial crop is decomposed into four localized anatomical zones: Forehead ($Z_1$), Periocular ($Z_2$), Mid-Face/Nose ($Z_3$), and Mouth/Jaw ($Z_4$), alongside the holistic full-face representation.
2. Patch Domain Adaptation: Dedicated neural adaptation heads project localized patch features into aligned, identity-discriminative latent representations, directly mitigating the domain gap of feeding isolated masked bands into a globally-trained CNN.
3. Physics-Informed Attention Fusion: An early feature-level neural attention layer fuses the 128-dimensional deep feature embeddings prior to distance calculation. Deterministic physical exposure scores ($E_i \in [0, 1]$) and anatomical prior weights ($P_i$) are directly injected into the attention logits as a physics prior and hard cutoff mask.
4. Zero-Trust Entropy Gate: If total active biometric entropy falls below $\text{MIN\_ENTROPY\_THRESHOLD} = 0.15$, the system abstains from making a match (INDETERMINATE), preventing fraudulent impostor false accepts under severe occlusions.

Pipeline
1. Automated Preprocessing: Canonical eye coordinates are extracted from bounding boxes, generating standardized frontal-aligned mugshot galleries and CCTV probes.
2. Continuous Exposure Estimation: Each zone receives a continuous physical exposure score ($E_i \in [0, 1]$) based on YCrCb skin-tone density (0.70) and high-frequency Sobel texture variance (0.30).
3. Domain Adaptation: Isolated patch embeddings are passed through trained domain adapters to align their feature distributions with whole-face biometric identity.
4. Physics-Informed Feature-Level Fusion: Probe and gallery embeddings are dynamically fused into unified 128-dimensional latent vectors via physics attention.
5. Verification: Direct cosine distance between the unified fused representations renders the biometric authentication decision.

Demonstration Tools
-------------------
- **Interactive Live Camera Face Matcher**:
  ```bash
  python demo/live_camera_matcher.py --model models/face_recognizer_fast.onnx --photo data/gallery/P1L_S1_C1.1_0.jpg
  ```
  Real-time webcam matching against reference photos, providing zone exposure HUD and comparative confidence scores for Holistic, Patch-Only, and Hybrid v6 models.

- **Synthetic Probe Visualizer**:
  ```bash
  python demo/live_visualizer.py --model models/face_recognizer_fast.onnx
  ```