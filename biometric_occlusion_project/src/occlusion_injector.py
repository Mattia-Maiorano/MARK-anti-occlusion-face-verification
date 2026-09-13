"""
occlusion_injector.py
---------------------
Synthetic Occlusion Augmentation Engine.

Applies four experimental degradation profiles to canonical 112 × 112 face
crops **in memory**, simulating real-world concealment accessories encountered
in crowd or protest-surveillance scenarios.  The occluded arrays are passed
directly into the Spatial Slicer without touching the filesystem.

Third-party : NumPy (array slicing and zeroing)
Custom      : Pixel-row zeroing boundaries for each of the 4 occlusion modes.

Occlusion schedule (all coordinates in 112 × 112 pixel space)
--------------------------------------------------------------
+------------------+-----------------------------+----------------------------+
| Mode             | Zeroed row range            | Anatomical simulation      |
+==================+=============================+============================+
| Baseline         | (none)                      | Unmodified probe crop      |
| Lower_Occlusion  | rows [56 : 112]             | Medical mask / scarf       |
| Upper_Occlusion  | rows [20 :  56]             | Sunglasses / low-brim cap  |
| Dual_Occlusion   | rows [20 :  56] ∪ [56 :112] | Mask + sunglasses combined |
+------------------+-----------------------------+----------------------------+
"""

from __future__ import annotations

import numpy as np

# ── Canonical image size enforced by this module ──────────────────────────────
CANONICAL_H = 112
CANONICAL_W = 112

# ── Occlusion band boundaries (row indices, 0-based, in 112 px height) ────────
_LOWER_START = 56    # rows [56 : 112]  → lower face (mask / scarf)
_UPPER_START = 20    # rows [20 :  56]  → eye / nose bridge region
_UPPER_END   = 56

# ── Public mode registry ──────────────────────────────────────────────────────
OCCLUSION_MODES: tuple[str, ...] = (
    "Baseline",
    "Lower_Occlusion",
    "Upper_Occlusion",
    "Dual_Occlusion",
)


def apply_occlusion(face_112: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply a single synthetic occlusion condition to a 112 × 112 BGR face crop.

    The input array is **never modified in-place**; a fresh copy is returned
    for every call.

    Parameters
    ----------
    face_112 : np.ndarray
        Canonical face crop, shape ``(112, 112, 3)``, dtype ``uint8``.
    mode : str
        One of the four mode strings in :data:`OCCLUSION_MODES`.

    Returns
    -------
    np.ndarray
        Occluded copy, same shape and dtype as the input.

    Raises
    ------
    ValueError
        If ``face_112`` is not exactly ``(112, 112, 3)`` or ``mode`` is
        not one of the recognised occlusion mode strings.
    """
    if face_112.shape != (CANONICAL_H, CANONICAL_W, 3):
        raise ValueError(
            f"[occlusion_injector] Expected shape ({CANONICAL_H}, {CANONICAL_W}, 3), "
            f"received {face_112.shape}.  Resize to canonical dimensions first."
        )
    if mode not in OCCLUSION_MODES:
        raise ValueError(
            f"[occlusion_injector] Unrecognised mode '{mode}'.  "
            f"Valid modes: {OCCLUSION_MODES}"
        )

    out = face_112.copy()

    if mode == "Baseline":
        pass  # return untouched copy

    elif mode == "Lower_Occlusion":
        # Zero out the lower 50 % of the face (rows 56 → 112)
        out[_LOWER_START:, :, :] = 0

    elif mode == "Upper_Occlusion":
        # Zero out the eye / nose-bridge band (rows 20 → 56)
        out[_UPPER_START:_UPPER_END, :, :] = 0

    elif mode == "Dual_Occlusion":
        # Both bands simultaneously (mask + sunglasses)
        out[_UPPER_START:_UPPER_END, :, :] = 0
        out[_LOWER_START:, :, :] = 0

    return out


def apply_all_conditions(face_112: np.ndarray) -> dict[str, np.ndarray]:
    """
    Generate all four occlusion conditions for a single face crop.

    Convenience wrapper that calls :func:`apply_occlusion` for every mode and
    returns them as an ordered mapping.

    Parameters
    ----------
    face_112 : np.ndarray
        Canonical ``(112, 112, 3)`` BGR face crop.

    Returns
    -------
    dict[str, np.ndarray]
        ``{ mode_name: occluded_image }`` for all four modes.
        The images are independent copies; mutating one does not affect others.

    Raises
    ------
    ValueError
        Propagated from :func:`apply_occlusion` on invalid input.
    """
    return {mode: apply_occlusion(face_112, mode) for mode in OCCLUSION_MODES}
