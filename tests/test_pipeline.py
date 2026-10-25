"""
test_pipeline.py
----------------
Unit Test Suite for the M.A.R.K. v5 Biometric Pipeline.

Covers all core modules in isolation:
    • src/occlusion_injector.py   (TestOcclusionInjector)
    • src/spatial_slicer.py       (TestSpatialSlicer)
    • src/occlusion_estimator.py  (TestOcclusionEstimator)
    • src/dynamic_matcher.py      (TestDynamicMatcherV5)
    • src/chokepoint_extractor.py (TestChokepointXMLParser)

Run with:
    pytest -v tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import cv2
import numpy as np
import pytest
import torch

# Ensure project root is importable when running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.occlusion_injector import (
    apply_occlusion,
    apply_all_conditions,
    OCCLUSION_MODES,
    CANONICAL_H,
    CANONICAL_W,
)
from src.spatial_slicer import (
    resize_to_canonical,
    slice_into_zones,
    ZONE_BOUNDS,
    CANONICAL_SIZE,
)
from src.occlusion_estimator import (
    compute_skin_density,
    compute_texture_variance,
    compute_exposure_score,
    compute_exposure_vector,
    MIN_EXPOSURE_THRESHOLD,
)
from src.dynamic_matcher import (
    cosine_distance,
    match,
    PatchDomainAdapter,
    DistanceScaler,
    match_dynamic_z_fusion,
    AdaptedLateScoreFusionModel,
    PATCH_PRIORS,
    HOLISTIC_PRIOR,
    MIN_ENTROPY_THRESHOLD,
    STATUS_VALID,
    STATUS_INDETERMINATE,
    EMBEDDING_DIM,
)
from src.chokepoint_extractor import (
    parse_chokepoint_xml,
    _select_best_frame,
    _sample_probe_frames,
)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def _make_face(seed: int = 42) -> np.ndarray:
    """Generate a reproducible 112 × 112 synthetic face-like BGR image."""
    rng = np.random.default_rng(seed)
    # YCrCb-compatible skin tone
    img = np.full((112, 112, 3), [90, 145, 195], dtype=np.uint8)
    # Eyes (dark, positioned in Zone 2 row band 28-56)
    cv2.ellipse(img, (34, 42), (13, 7), 0, 0, 360, (25, 25, 25), -1)
    cv2.ellipse(img, (78, 42), (13, 7), 0, 0, 360, (25, 25, 25), -1)
    # Mild noise
    noise = rng.integers(0, 18, img.shape, dtype=np.uint8)
    return np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def face_112():
    return _make_face(seed=42)


def _l2_vec(seed: int = 0, dim: int = EMBEDDING_DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — OCCLUSION INJECTOR
# ─────────────────────────────────────────────────────────────────────────────

class TestOcclusionInjector:
    def test_output_shape_and_dtype(self, face_112):
        for mode in OCCLUSION_MODES:
            out = apply_occlusion(face_112, mode)
            assert out.shape == (112, 112, 3)
            assert out.dtype == np.uint8

    def test_baseline_is_identical(self, face_112):
        out = apply_occlusion(face_112, "Baseline")
        np.testing.assert_array_equal(out, face_112)

    def test_lower_occlusion_modifies_lower_rows(self, face_112):
        out = apply_occlusion(face_112, "Lower_Occlusion")
        assert not np.array_equal(out[56:, :, :], face_112[56:, :, :])
        np.testing.assert_array_equal(out[:56, :, :], face_112[:56, :, :])

    def test_upper_occlusion_modifies_upper_rows(self, face_112):
        out = apply_occlusion(face_112, "Upper_Occlusion")
        assert not np.array_equal(out[28:56, :, :], face_112[28:56, :, :])
        np.testing.assert_array_equal(out[56:, :, :], face_112[56:, :, :])

    def test_dual_occlusion_modifies_both(self, face_112):
        out = apply_occlusion(face_112, "Dual_Occlusion")
        assert not np.array_equal(out[28:56, :, :], face_112[28:56, :, :])
        assert not np.array_equal(out[56:, :, :], face_112[56:, :, :])

    def test_invalid_mode_raises(self, face_112):
        with pytest.raises(ValueError):
            apply_occlusion(face_112, "Unknown_Mode")

    def test_apply_all_conditions_returns_all_keys(self, face_112):
        all_c = apply_all_conditions(face_112)
        assert set(all_c.keys()) == set(OCCLUSION_MODES)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — SPATIAL SLICER
# ─────────────────────────────────────────────────────────────────────────────

class TestSpatialSlicer:
    def test_resize_to_canonical_preserves_shape(self):
        img = np.zeros((200, 150, 3), dtype=np.uint8)
        resized = resize_to_canonical(img)
        assert resized.shape == (112, 112, 3)

    def test_slice_into_zones_returns_4_zones(self, face_112):
        zones = slice_into_zones(face_112)
        assert len(zones) == 4
        for z in zones:
            assert z.shape == (112, 112, 3)

    def test_slice_union_reconstructs_original(self, face_112):
        zones = slice_into_zones(face_112)
        reconstructed = np.zeros_like(face_112)
        for r_start, r_end in ZONE_BOUNDS:
            canvas = np.zeros_like(face_112)
        for z in zones:
            reconstructed += z
        np.testing.assert_array_equal(reconstructed, face_112)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — OCCLUSION ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

class TestOcclusionEstimator:
    def test_skin_density_range(self, face_112):
        zones = slice_into_zones(face_112)
        for i in range(len(ZONE_BOUNDS)):
            d = compute_skin_density(zones[i])
            assert 0.0 <= d <= 1.0

    def test_exposure_vector_shape_and_range(self, face_112):
        zones = slice_into_zones(face_112)
        vec = compute_exposure_vector(zones)
        assert vec.shape == (4,)
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — M.A.R.K. v5.1 DYNAMIC MATCHER (Adapted Late Score Fusion)
# ─────────────────────────────────────────────────────────────────────────────

class TestDynamicMatcherV51:
    def test_cosine_distance_properties(self):
        v1 = _l2_vec(1)
        v2 = _l2_vec(2)
        # Identical vectors -> 0.0
        assert cosine_distance(v1, v1) < 1e-5
        # Orthogonal vectors -> 1.0
        e1 = np.zeros(128); e1[0] = 1.0
        e2 = np.zeros(128); e2[1] = 1.0
        assert abs(cosine_distance(e1, e2) - 1.0) < 1e-5
        # Symmetry
        assert abs(cosine_distance(v1, v2) - cosine_distance(v2, v1)) < 1e-6
        # Range
        assert 0.0 <= cosine_distance(v1, v2) <= 2.0
        # Zero vector safety
        assert cosine_distance(np.zeros(128), v1) == 1.0

    def test_patch_domain_adapter(self):
        adapter = PatchDomainAdapter(dim=128)
        x = torch.randn(10, 128)
        out = adapter(x)
        assert out.shape == (10, 128)
        norms = torch.norm(out, p=2, dim=-1)
        np.testing.assert_allclose(norms.detach().numpy(), np.ones(10), atol=1e-5)

    def test_distance_scaler(self):
        scaler = DistanceScaler()
        assert not scaler.is_fitted
        data = np.random.randn(50, 5) * 0.2 + 0.8
        scaler.fit(data)
        assert scaler.is_fitted
        assert scaler.mean.shape == (5,)
        assert scaler.scale.shape == (5,)
        transformed = scaler.transform(data[0])
        assert transformed.shape == (5,)

    def test_match_dynamic_z_fusion_indeterminate(self):
        scaler = DistanceScaler()
        data = np.random.randn(50, 5) * 0.2 + 0.8
        scaler.fit(data)

        # Zero exposure
        raw_5d = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        zero_exp = np.zeros(4)
        res = match_dynamic_z_fusion(raw_5d, zero_exp, scaler)
        assert res["status"] == STATUS_INDETERMINATE
        assert res["distance"] is None

    def test_match_dynamic_z_fusion_valid(self):
        scaler = DistanceScaler()
        data = np.random.randn(50, 5) * 0.2 + 0.8
        scaler.fit(data)

        raw_5d = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        exp = np.array([1.0, 1.0, 1.0, 1.0])
        res = match_dynamic_z_fusion(raw_5d, exp, scaler)
        assert res["status"] == STATUS_VALID
        assert res["distance"] is not None
        assert isinstance(res["distance"], float)

    def test_adapted_late_score_fusion_model(self):
        model = AdaptedLateScoreFusionModel(dim=128)
        # Dummy data
        train_data = np.random.randn(20, 5) * 0.2 + 0.8
        model.fit_scaler(train_data)

        probe_zones = [_l2_vec(i) for i in range(4)]
        probe_hol = _l2_vec(10)
        gal_zones = [_l2_vec(i) for i in range(4)]
        gal_hol = _l2_vec(10)
        exp = np.array([1.0, 1.0, 1.0, 1.0])

        res = model.match(
            probe_zones=probe_zones,
            probe_holistic=probe_hol,
            probe_exposure=exp,
            gallery_zones=gal_zones,
            gallery_holistic=gal_hol,
            mode="hybrid",
        )
        assert res["status"] == STATUS_VALID
        assert res["distance"] is not None

    def test_learned_gating_network_v6(self):
        from src.dynamic_matcher import LearnedGatingNetwork
        net = LearnedGatingNetwork(in_dim=9, hidden_dim=16, out_dim=5)
        # Verify parameter count strictly < 500
        n_params = sum(p.numel() for p in net.parameters())
        assert n_params < 500
        assert n_params == 245

        # Forward pass batch of 4
        x9 = torch.randn(4, 9)
        exp4 = torch.rand(4, 4)
        w = net(x9, exp4)
        assert w.shape == (4, 5)
        # Weights must be non-negative and sum to 1.0
        assert torch.all(w >= 0.0)
        assert torch.allclose(torch.sum(w, dim=-1), torch.ones(4), atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — CHOKEPOINT XML PARSER
# ─────────────────────────────────────────────────────────────────────────────

class TestChokepointXMLParser:
    def _write_xml(self, content: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
            f.write(content)
            return f.name

    def test_parse_valid_xml(self):
        xml = """<?xml version="1.0" encoding="utf-8"?>
        <annotation>
            <frame number="1">
                <person id="0001">
                    <leftEye x="30" y="40"/>
                    <rightEye x="60" y="40"/>
                </person>
            </frame>
        </annotation>"""
        path = self._write_xml(xml)
        result = parse_chokepoint_xml(path)
        os.unlink(path)
        assert "0001" in result
        assert len(result["0001"]) == 1

    def test_select_best_frame(self):
        dets = [
            {"frame": 1, "iod": 30.0, "tilt": 5.0, "left_eye": (0,0), "right_eye": (30,5)},
            {"frame": 2, "iod": 55.0, "tilt": 0.5, "left_eye": (0,0), "right_eye": (55,0.5)},
        ]
        best = _select_best_frame(dets)
        assert best["frame"] == 2

    def test_sample_probe_frames(self):
        dets = [{"frame": i, "iod": 30.0, "tilt": 1.0, "left_eye": (0,0), "right_eye": (30,1)} for i in range(10)]
        sampled = _sample_probe_frames(dets, best_frame_num=0, n=4)
        assert len(sampled) == 4
        assert all(d["frame"] != 0 for d in sampled)
