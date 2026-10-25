"""
spatial_slicer.py
-----------------
Anatomical 4-Zone Face Partitioning Engine — Masking Implementation.

Decomposes a 112 × 112 canonical face image into four full-resolution
**masked** sub-images.  Each output retains the original pixel values only
within its designated anatomical row band; all other rows are set to zero
(black).  The 112 × 112 spatial context is preserved so that the frozen
FaceRecognizerSF backbone can process each zone image without resize.

Zone definitions (fixed for 112 × 112 input)
---------------------------------------------
+--------+-----------------------+-----------+---------------------------------+
| Zone   | Name                  | Row range | Proportional extent             |
+========+=======================+===========+=================================+
| Zone 1 | Forehead / Upper Brow | [  0 : 28)| 0.00 H → 0.25 H                |
| Zone 2 | Periocular (Eyes)     | [ 28 : 56)| 0.25 H → 0.50 H                |
| Zone 3 | Nose / Mid-Cheeks     | [ 56 : 78)| 0.50 H → 0.70 H (~0.696)       |
| Zone 4 | Mouth / Jaw           | [ 78 :112)| 0.70 H → 1.00 H                |
+--------+-----------------------+-----------+---------------------------------+

Third-party : OpenCV (resize), NumPy (array masking)
Custom      : Zone boundary definitions, masked-copy construction logic.
"""

from __future__ import annotations

import numpy as np
import cv2

# ── Canonical SFace input size ────────────────────────────────────────────────
CANONICAL_SIZE = (112, 112)   # (width, height) for cv2.resize

# ── Zone boundary table (start_row_inclusive, end_row_exclusive) ──────────────
ZONE_BOUNDS: list[tuple[int, int]] = [
    (0,  28),    # Zone 1: Forehead / Upper Brow
    (28, 56),    # Zone 2: Periocular — Eyes & Bridge
    (56, 78),    # Zone 3: Nose & Mid-Cheeks
    (78, 112),   # Zone 4: Mouth, Chin & Jawline
]

ZONE_NAMES: list[str] = [
    "Forehead",
    "Periocular",
    "Nose/Cheeks",
    "Mouth/Jaw",
]

_N_ZONES = len(ZONE_BOUNDS)


def resize_to_canonical(img: np.ndarray) -> np.ndarray:
    """
    Resize an arbitrary BGR face crop to the 112 × 112 SFace canonical size.

    Uses ``cv2.INTER_AREA`` for downscaling (best anti-aliasing) and
    ``cv2.INTER_LINEAR`` for upscaling.

    Parameters
    ----------
    img : np.ndarray
        Input BGR face crop, shape ``(H, W, 3)``, any ``H``, ``W ≥ 20``.

    Returns
    -------
    np.ndarray
        Resized image of shape ``(112, 112, 3)``, dtype preserved.

    Raises
    ------
    ValueError
        If ``img`` is not a 3-channel image or is smaller than 20 × 20 pixels.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"[spatial_slicer] Expected a 3-channel BGR image, got shape {img.shape}."
        )
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        raise ValueError(
            f"[spatial_slicer] Input image too small ({h} × {w}).  Minimum: 20 × 20."
        )

    interp = cv2.INTER_AREA if (h > 112 or w > 112) else cv2.INTER_LINEAR
    return cv2.resize(img, CANONICAL_SIZE, interpolation=interp)


def slice_into_zones(face_112: np.ndarray) -> list[np.ndarray]:
    """
    Partition a 112 × 112 face image into four anatomical masked sub-images.

    Implementation
    --------------
    For each zone ``i`` with row range ``[r_start, r_end)``:

    1. Allocate a zero-filled ``(112, 112, 3)`` canvas.
    2. Copy ``face_112[r_start:r_end, :, :]`` into the same rows of the canvas.
    3. Return the canvas as zone ``i``.

    This masking strategy (as opposed to cropping) preserves the full 112 × 112
    spatial layout required by ``FaceRecognizerSF.feature()``, while ensuring
    that only zone-specific pixels contribute to the embedding.

    Parameters
    ----------
    face_112 : np.ndarray
        Canonical face image of shape ``(112, 112, 3)``.

    Returns
    -------
    list[np.ndarray]
        List of ``_N_ZONES`` (4) masked BGR images, each ``(112, 112, 3)``.
        The union of all zone masks exactly reconstructs the input.

    Raises
    ------
    ValueError
        If ``face_112`` is not exactly ``(112, 112, 3)``.
        Call :func:`resize_to_canonical` before this function.
    """
    if face_112.shape != (112, 112, 3):
        raise ValueError(
            f"[spatial_slicer] Expected (112, 112, 3), received {face_112.shape}.  "
            "Call resize_to_canonical() before slice_into_zones()."
        )

    zones: list[np.ndarray] = []
    for r_start, r_end in ZONE_BOUNDS:
        canvas = np.zeros_like(face_112)
        canvas[r_start:r_end, :, :] = face_112[r_start:r_end, :, :]
        zones.append(canvas)

    return zones
