"""
occlusion_estimator.py
-----------------------
Patch-Level Occlusion Assessment & Continuous Exposure Estimation Engine.

For each of the 4 anatomical zone images produced by the Spatial Slicer, this
module computes a continuous Exposure Score  E_i ∈ [0.0, 1.0]  using two
independent, complementary image cues:

    Cue 1 — Skin-pixel chromaticity density (weight = 0.70)
        Fraction of active pixels that fall within the human skin locus in
        the YCrCb color space.  Skin bounds:
            Y  ∈ [  0, 255]   (luminance unrestricted — all complexions)
            Cr ∈ [133, 173]
            Cb ∈ [ 77, 127]

    Cue 2 — Gradient texture variance (weight = 0.30)
        Variance of Sobel edge magnitudes across the active region, normalised
        to [0, 1] by a ceiling constant.  Natural skin produces moderate,
        spatially distributed gradients; flat synthetic occluders (fabric,
        plastic lenses) concentrate gradients only at their edges, yielding
        distinctly different variance signatures.

Clamping rule
-------------
    E_i = 0.0   if   raw_score < 0.20

This hard clamp makes the downstream matcher treat effectively-occluded
patches as providing zero biometric information.

Active-region extraction
------------------------
Zone images are 112 × 112 with all-zero rows outside the zone band.  Both
cues are computed exclusively on the non-zero rows to avoid bias from the
zero-padded background.

Third-party : OpenCV (cvtColor, Sobel), NumPy (masking, statistics)
Custom      : YCrCb skin locus bounds, cue fusion formula, active-region
              extractor, clamping logic, exposure-vector assembler.
"""

from __future__ import annotations

import numpy as np
import cv2

# ── Skin chromaticity bounds in YCrCb space ───────────────────────────────────
_SKIN_Y_MIN,  _SKIN_Y_MAX  = 0,   255
_SKIN_CR_MIN, _SKIN_CR_MAX = 133, 173
_SKIN_CB_MIN, _SKIN_CB_MAX = 77,  127

# ── Score fusion weights ──────────────────────────────────────────────────────
_SKIN_WEIGHT    = 0.70
_TEXTURE_WEIGHT = 0.30

# ── Exposure threshold & texture normalisation ceiling ────────────────────────
MIN_EXPOSURE_THRESHOLD   = 0.20
_TEXTURE_VAR_CEILING     = 3000.0   # variance ≥ this → normalised to 1.0
_TEXTURE_VAR_CEILING_INV = 1.0 / _TEXTURE_VAR_CEILING


def _extract_active_rows(zone_img: np.ndarray) -> np.ndarray | None:
    """
    Return only the non-zero rows from a masked zone image.

    Zone images produced by :mod:`spatial_slicer` have zeros in every row
    outside the anatomical band.  This function recovers the raw band pixels
    for unbiased chromaticity and texture statistics.

    Parameters
    ----------
    zone_img : np.ndarray
        Full ``(112, 112, 3)`` masked zone image, dtype ``uint8``.

    Returns
    -------
    np.ndarray or None
        Sub-array of only the active rows, shape ``(n_rows, 112, 3)``.
        Returns ``None`` if every pixel is zero (zone is fully occluded).
    """
    # A row is "active" if at least one pixel across all 3 channels is non-zero
    row_has_signal = np.any(zone_img > 0, axis=(1, 2))
    if not np.any(row_has_signal):
        return None
    return zone_img[row_has_signal]   # shape: (n_active, 112, 3)


def compute_skin_density(patch: np.ndarray) -> float:
    """
    Compute the fraction of pixels within the human skin chromaticity locus.

    The YCrCb color space is used because Cr and Cb are largely illumination-
    invariant, making the skin locus compact and consistent across ethnicities
    and lighting conditions.

    Parameters
    ----------
    patch : np.ndarray
        BGR sub-image of arbitrary size, dtype ``uint8``.

    Returns
    -------
    float
        Skin-pixel density ∈ [0.0, 1.0].
    """
    if patch is None or patch.size == 0:
        return 0.0

    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
    Y  = ycrcb[:, :, 0].astype(np.int32)
    Cr = ycrcb[:, :, 1].astype(np.int32)
    Cb = ycrcb[:, :, 2].astype(np.int32)

    skin_mask = (
        (Y  >= _SKIN_Y_MIN)  & (Y  <= _SKIN_Y_MAX)  &
        (Cr >= _SKIN_CR_MIN) & (Cr <= _SKIN_CR_MAX)  &
        (Cb >= _SKIN_CB_MIN) & (Cb <= _SKIN_CB_MAX)
    )

    n_total = skin_mask.size
    return float(np.sum(skin_mask)) / n_total if n_total > 0 else 0.0


def compute_texture_variance(patch: np.ndarray) -> float:
    """
    Compute the normalised variance of Sobel gradient magnitudes.

    Sobel operators on both axes are combined as:
        ``magnitude = sqrt(Gx² + Gy²)``

    The variance is normalised by ``_TEXTURE_VAR_CEILING`` and clamped to 1.0.

    Parameters
    ----------
    patch : np.ndarray
        BGR sub-image, dtype ``uint8``.

    Returns
    -------
    float
        Normalised texture variance ∈ [0.0, 1.0].
    """
    if patch is None or patch.size == 0:
        return 0.0

    gray    = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    raw_var = float(np.var(magnitude))
    return min(raw_var * _TEXTURE_VAR_CEILING_INV, 1.0)


def compute_exposure_score(zone_img: np.ndarray) -> float:
    """
    Derive the continuous Exposure Score ``E_i ∈ [0.0, 1.0]`` for one zone.

    Formula::

        raw_score = 0.70 × skin_density + 0.30 × texture_variance
        E_i       = 0.0   if  raw_score < 0.20
                  = raw_score  otherwise

    Parameters
    ----------
    zone_img : np.ndarray
        Full ``(112, 112, 3)`` masked zone image.

    Returns
    -------
    float
        Exposure score, clamped to ``0.0`` when below :data:`MIN_EXPOSURE_THRESHOLD`.
    """
    active = _extract_active_rows(zone_img)

    if active is None:
        return 0.0

    skin    = compute_skin_density(active)
    texture = compute_texture_variance(active)

    raw = _SKIN_WEIGHT * skin + _TEXTURE_WEIGHT * texture
    return 0.0 if raw < MIN_EXPOSURE_THRESHOLD else float(raw)


def compute_exposure_vector(zones: list[np.ndarray]) -> np.ndarray:
    """
    Compute the 4-dimensional Exposure Vector for a complete facial image.

    Parameters
    ----------
    zones : list[np.ndarray]
        Exactly 4 masked ``(112, 112, 3)`` zone images — output of
        :func:`spatial_slicer.slice_into_zones`.

    Returns
    -------
    np.ndarray
        Shape ``(4,)``, dtype ``float32``.  Each element ``E_i ∈ [0.0, 1.0]``.

    Raises
    ------
    ValueError
        If ``zones`` does not contain exactly 4 elements.
    """
    if len(zones) != 4:
        raise ValueError(
            f"[occlusion_estimator] Expected 4 zone images, received {len(zones)}."
        )
    return np.array(
        [compute_exposure_score(z) for z in zones],
        dtype=np.float32,
    )
