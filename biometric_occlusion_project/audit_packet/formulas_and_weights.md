# M.A.R.K. Biometric Architecture: Weights & Mathematical Formulas
**Mugshot Assessment & Recognition Kernel — v4 (Dynamic Normalized Z-Fusion)**

---

## 1. System Overview
M.A.R.K. is a biometric verification engine engineered for degraded surveillance conditions (e.g., subjects wearing masks or sunglasses). Rather than relying purely on an unweighted holistic CNN, M.A.R.K. slices the canonical face ($112 \times 112$) into four anatomical zones, assesses physical skin/texture exposure, normalizes distances to prevent scale distortion, and dynamically re-distributes decision authority.

---

## 2. Anatomical Patch Prior Weights (\(W_{patch}\) and \(W_{holistic}\))
Intrinsic biometric discriminability priors define the baseline information density of each anatomical zone:

| Region | Zone Identifier | Prior Weight (\(W_i\)) | Anatomical Rationale |
|---|---|---|---|
| **Forehead** | Zone 1 (\(Z_1\)) | **0.05** | Low individuality; often covered by hair, hats, or frontal glare. |
| **Periocular** | Zone 2 (\(Z_2\)) | **0.55** | **Highest authority**; captures iris, inter-ocular distance, brow bone structure. |
| **Mid-Face / Nose** | Zone 3 (\(Z_3\)) | **0.15** | Moderate individuality; malar structure and nasal bridge. |
| **Mouth / Jaw** | Zone 4 (\(Z_4\)) | **0.25** | Moderate-high individuality; lip geometry, chin contour, philtrum. |
| **Global Holistic** | Holistic (\(H\)) | **1.00** | Full-face baseline representation extracted by SFace CNN. |

- Base Patch Priors: `W_patch = [0.05, 0.55, 0.15, 0.25]`
- Holistic Prior: `W_holistic = 1.0` (Hybrid) or `0.0` (Patch-Only)
- Zero-Trust Entropy Gate Threshold: `MIN_ENTROPY_THRESHOLD = 0.15`

---

## 3. Mathematical Formulas

### 3.1 Cosine Distance
For any pair of L2-normalized feature embeddings \(v_1, v_2\):
$$\text{dist}(v_1, v_2) = 1.0 - \frac{v_1 \cdot v_2}{\|v_1\| \|v_2\|}$$

### 3.2 Continuous Exposure Estimation (\(E_i\))
For each anatomical patch \(i \in \{1, 2, 3, 4\}\):
1. **Skin-Tone Density (\(S_i\))**: Proportion of pixels satisfying human skin thresholds in the YCrCb color space:
   $$Cr \in [133, 173], \quad Cb \in [77, 127]$$
2. **Texture Variance (\(T_i\))**: High-frequency gradient variance calculated via Sobel operator (clamped at 500.0):
   $$T_i = \min\left(1.0, \frac{\text{Var}(\nabla I)}{500.0}\right)$$
3. **Zone Exposure Score**:
   $$E_i = 0.70 \cdot S_i + 0.30 \cdot T_i$$
   *(Hard floor: If \(E_i < 0.20\), then \(E_i \leftarrow 0.0\))*.
4. **Holistic Exposure Mean**:
   $$\bar{E} = \frac{1}{4} \sum_{i=1}^4 E_i$$

### 3.3 Dynamic Z-Score Normalization
To prevent scale disparity between localized patch distances and global whole-face cosine distances, raw distances are transformed into dimensionless Z-scores using parameters estimated strictly on the training partition:
$$z_i = \frac{d_i - \mu_i}{\sigma_i}, \quad \text{for } i \in \{1, 2, 3, 4, \text{holistic}\}$$

**Fitted Scaler Parameters (from 60% Training Split):**
- Means (\(\mu\)):
  - \(\mu_1\) (Forehead) = `0.6278`
  - \(\mu_2\) (Periocular) = `0.9293`
  - \(\mu_3\) (Mid-face) = `0.8793`
  - \(\mu_4\) (Mouth/Jaw) = `0.7703`
  - \(\mu_{hol}\) (Holistic) = `0.9566`
- Scales (\(\sigma\)):
  - \(\sigma_1\) (Forehead) = `0.1820`
  - \(\sigma_2\) (Periocular) = `0.1613`
  - \(\sigma_3\) (Mid-face) = `0.2254`
  - \(\sigma_4\) (Mouth/Jaw) = `0.2439`
  - \(\sigma_{hol}\) (Holistic) = `0.1305`

### 3.4 Active Authority and Dynamic Re-Normalizing Fusion
1. **Active Exposure-Weighted Authority**:
   $$\Omega_{\text{active}} = W_{holistic} \cdot \bar{E} + \sum_{i=1}^4 (W_i \cdot E_i)$$
2. **Zero-Trust Fail-Safe (Safety Gate)**:
   $$\text{If } \Omega_{\text{active}} < 0.15 \implies \text{Verdict: } \textbf{INDETERMINATE\_INSUFFICIENT\_INFORMATION}$$
   *(The system refuses to match rather than guessing blindly on extreme occlusions).*
3. **Dynamic Fused Distance**:
   $$Z_{\text{fused}} = \frac{W_{holistic} \cdot \bar{E} \cdot z_{hol} + \sum_{i=1}^4 (W_i \cdot E_i \cdot z_i)}{\Omega_{\text{active}}}$$
4. **Normalized Dynamic Fusion Weights**:
   $$w_i = \frac{W_i \cdot E_i}{\Omega_{\text{active}}}, \quad w_{hol} = \frac{W_{holistic} \cdot \bar{E}}{\Omega_{\text{active}}}$$

---

## 4. Operational Decision Rule
Given operational Equal Error Rate (EER) threshold \(\tau\):
- **MATCH (ACCEPT)**: \(Z_{\text{fused}} \le \tau\)
- **NO MATCH (REFUSE)**: \(Z_{\text{fused}} > \tau\) or \(\text{Status} = \text{INDETERMINATE}\)

**Operational Thresholds by Condition (Tri-System Benchmark):**
- Baseline: `\tau = -1.6447`
- Lower Occlusion (Mask): `\tau = -1.5088`
- Upper Occlusion (Sunglasses): `\tau = -1.0769`
- Dual Occlusion: `\tau = 0.2298` *(Dual Occlusion triggers 100% FTA gate rejection)*
