"""
embedder.py
-----------
Deep Patch Feature Extraction via Frozen SFace Backbone.

Wraps OpenCV's ``cv2.FaceRecognizerSF`` to provide zone-level and holistic
128-dimensional L2-normalised feature vectors.  The ONNX backbone weights are
never modified; all occlusion adaptation is performed at the metric-fusion layer.

Third-party : OpenCV ``FaceRecognizerSF`` (ONNX inference engine)
Custom      : Fail-fast model validation, L2-normalisation enforcement,
              zone-batch embedding pipeline, holistic baseline helper.

Fail-fast policy
----------------
If the ONNX model file is absent from disk, :class:`FaceEmbedder.__init__`
raises :class:`FileNotFoundError` immediately with the exact ``curl`` /
``wget`` command needed to obtain the file.  There is no mock or PCA fallback.

Model source
------------
    URL  : https://github.com/opencv/opencv_zoo/raw/main/models/
           face_recognition_sface/face_recognition_sface_2021dec.onnx
    Save : models/face_recognizer_fast.onnx
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# ── Model constants ───────────────────────────────────────────────────────────
SFACE_DOWNLOAD_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
EMBEDDING_DIM = 128


class FaceEmbedder:
    """
    Frozen SFace feature extractor.

    Loads the pre-trained SFace ONNX backbone via ``cv2.FaceRecognizerSF``
    and exposes methods for patch-level (zone) and full-face (holistic) embedding.

    Parameters
    ----------
    model_path : str
        Filesystem path to ``face_recognizer_fast.onnx``.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist.  Download instructions are included
        in the exception message.
    RuntimeError
        If OpenCV fails to initialise the ONNX model after locating the file.
    """

    def __init__(self, model_path: str) -> None:
        # ── Strict existence check — no fallback ─────────────────────────────
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"\n[embedder] SFace ONNX model not found at: '{model_path}'\n\n"
                "  Download the model with one of the following commands:\n\n"
                f"    curl -L '{SFACE_DOWNLOAD_URL}' \\\n"
                f"         -o '{model_path}'\n\n"
                f"    wget -O '{model_path}' \\\n"
                f"         '{SFACE_DOWNLOAD_URL}'\n\n"
                "  Then re-run the pipeline.\n"
            )

        print(f"[embedder] Loading SFace backbone: {model_path}")
        try:
            # OpenCV ≥ 4.5.4 / 5.x  —  second argument is config path (empty for ONNX)
            self._recognizer = cv2.FaceRecognizerSF.create(model_path, "")
        except cv2.error as exc:
            raise RuntimeError(
                f"[embedder] cv2.FaceRecognizerSF.create() failed: {exc}\n"
                "Ensure opencv-contrib-python ≥ 4.5.4 is installed and the ONNX "
                "file is a valid SFace model."
            ) from exc

        print(f"[embedder] Backbone ready.  Embedding dim: {EMBEDDING_DIM}.")

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_patch(self, patch_112: np.ndarray) -> np.ndarray:
        """
        Extract an L2-normalised feature vector from a single 112 × 112 image.

        The image may be a masked zone image (with zero rows outside the active
        band) or a full unmasked face crop.  The backbone processes the entire
        112 × 112 spatial context in both cases.

        Parameters
        ----------
        patch_112 : np.ndarray
            BGR image of shape ``(112, 112, 3)``, dtype ``uint8``.

        Returns
        -------
        np.ndarray
            L2-normalised embedding, shape ``(128,)``, dtype ``float32``.
            A zero vector is returned if the backbone produces a near-zero output
            (fully blank zone — no information available).

        Raises
        ------
        ValueError
            If ``patch_112`` does not have shape ``(112, 112, 3)``.
        """
        if patch_112.shape != (112, 112, 3):
            raise ValueError(
                f"[embedder] embed_patch requires (112, 112, 3), got {patch_112.shape}."
            )

        # ``feature()`` expects a uint8 BGR image aligned to 112 × 112
        raw_feat = self._recognizer.feature(patch_112)

        # Flatten: output may be (1, 128) or (128,) depending on OpenCV version
        vec = np.array(raw_feat, dtype=np.float32).flatten()

        # Enforce L2 normalisation
        norm = float(np.linalg.norm(vec))
        if norm > 1e-10:
            vec = vec / norm
        else:
            # Completely blank zone → zero embedding (signals no information)
            vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        return vec

    def embed_zones(self, zones: list[np.ndarray]) -> list[np.ndarray]:
        """
        Embed all 4 anatomical zone images.

        Parameters
        ----------
        zones : list[np.ndarray]
            4-element list of masked ``(112, 112, 3)`` zone images produced by
            :func:`spatial_slicer.slice_into_zones`.

        Returns
        -------
        list[np.ndarray]
            4 L2-normalised ``(128,)`` embedding vectors.

        Raises
        ------
        ValueError
            If ``zones`` does not contain exactly 4 elements.
        """
        if len(zones) != 4:
            raise ValueError(
                f"[embedder] embed_zones requires exactly 4 zones, got {len(zones)}."
            )
        return [self.embed_patch(z) for z in zones]

    def embed_holistic(self, face_112: np.ndarray) -> np.ndarray:
        """
        Embed the full 112 × 112 unmasked face for holistic baseline comparison.

        Parameters
        ----------
        face_112 : np.ndarray
            Full face image ``(112, 112, 3)``, dtype ``uint8``.

        Returns
        -------
        np.ndarray
            L2-normalised 128-dim embedding vector.
        """
        return self.embed_patch(face_112)
