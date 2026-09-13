M.A.R.K. — Mugshot Assessment & Recognition Kernel
Final Project Report & Results Summary (v6 Architecture)

1. Executive Summary
M.A.R.K. is a robust, occlusion-aware biometric verification system designed to solve a critical vulnerability in modern surveillance: the severe degradation of facial recognition accuracy when subjects present with partial facial occlusions (e.g., medical masks, sunglasses, or heavy winter clothing).
By transitioning from rigid, hand-crafted prior heuristics to a Learned Score-Level Gating Network, M.A.R.K. v6 successfully mitigates the catastrophic failure modes of traditional deep CNNs. The final architecture systematically surpasses the unweighted baseline CNN on clear faces, masked faces, and faces with sunglasses, while introducing a mathematically audited adaptive security gate to prevent spoofing under extreme disguise.

2. Project Goal
Standard Deep Convolutional Neural Networks (CNNs) achieve near-perfect accuracy on clean faces but rely heavily on global geometric relationships. When large portions of the face are hidden, this global reliance causes severe accuracy regressions. The primary objective of M.A.R.K. was to build a system that securely matches pristine enrolled mugshots against degraded, uncooperative CCTV captures. By breaking the face down into localized anatomical zones and mathematically routing verification authority strictly to visible features, the system aims to maintain high-confidence authentication despite severe visual degradation.

3. Pipeline & Architectural Evolution
The M.A.R.K. pipeline wraps a frozen off-the-shelf CNN (FaceRecognizerSF) in a highly secure, mathematically auditable decision layer. Its evolution culminated in the v6 architecture:
⚬ Automated Preprocessing: Ingests bounding boxes and extracts canonical eye-coordinates, standardizing gallery mugshots and uncooperative probes.
⚬ Continuous Exposure Estimation: Analyzes four anatomical zones (Forehead, Periocular, Mid-Face, Mouth/Jaw) using a YCrCb skin-density and Sobel texture variance heuristic to calculate a continuous physical exposure score (E_i \in [0, 1]).
⚬ Patch Domain Adapters & Z-Score Normalization: SFace embeddings extracted from isolated patches undergo specialized 128-dimensional neural adaptations to align them into a single identity latent space. Raw cosine distances are then normalized into dimensionless Z-scores to prevent scale mismatch between local and global inferences.
⚬ Learned Score-Level Dynamic Gating: After early experiments with high-dimensional Cross-Attention (which failed due to identity-starvation and vector interference on a small dataset), the system adopted a highly efficient 2-layer MLP (245 parameters). This network evaluates 9 continuous scalars (5 Z-scores and 4 exposure values) and learns the optimal non-linear authority weights, routing decision power safely and smoothly.
⚬ Calibrated Adaptive Confidence Gate: Rather than instituting a blanket 100% rejection rate for extreme (Dual) occlusions, an optimized entropy threshold (\tau_{\text{gate}} = 0.0350) dynamically admits valid partial-face probes while actively blocking high-entropy impostors.
4. Final Verification Metrics (v6 Benchmarks)
The v6 system was benchmarked against the standalone Holistic CNN baseline using a two-pass evaluation protocol. M.A.R.K. v6 systematically outperformed the standalone baseline across all major operational vectors:
Occlusion Condition	Holistic Baseline EER	M.A.R.K. v6 Hybrid EER	Holistic AUC	M.A.R.K. v6 Hybrid AUC
Baseline (Clean Face)	2.33%	1.67%	0.9983	0.9987
Lower Occlusion (Mask)	4.00%	3.83%	0.9879	0.9926
Upper Occlusion (Sunglasses)	10.00%	8.00%	0.9676	0.9766
Key Breakthroughs:
⚬ Lower Occlusion Barrier Broken: By using learned score-level gating, the system finally surpassed the strict 4.00% mask baseline, achieving a 3.83% Equal Error Rate (EER) while posting an exceptional 0.9926 Area Under the Curve (AUC).
⚬ Baseline Enhancement: Applying localized patch scrutiny actually improved the overall biometric verification of clean, unoccluded faces (dropping EER from 2.33% to 1.67%).
5. Security & Dual-Occlusion Forensics
Traditional CNNs completely collapse under Dual Occlusion (mask + sunglasses), randomly guessing with a 42.00% EER. Earlier iterations of M.A.R.K. relied on a "Zero-Trust" hard gate that rejected 100% of these attempts (100% FTA rate).
The v6 Adaptive Gate completely recovered operational utility:
⚬ The Failure to Acquire (FTA) rate was reduced from 100% to 68.6%.
⚬ The system safely admitted and matched valid biometric signals, establishing a highly usable 18.18% EER (0.9011 AUC) on heavily disguised individuals.
⚬ Crucially, despite lowering the rejection floor, the system safely intercepted and prevented 485 fraudulent impostor attempts that would have otherwise breached the security perimeter.
6. Conclusion
M.A.R.K. v6 successfully demonstrates that combining deterministic physical heuristics (exposure estimation) with ultra-lightweight neural gating (low-dimensional score fusion) produces a biometric architecture far more resilient than standard global convolutions. By correctly diagnosing the data-limitations of the underlying dataset and abandoning high-dimensional vector fusion in favor of scalar distance weighting, the project established a highly accurate, mathematically auditable, and operationally secure facial recognition framework for degraded surveillance environments.