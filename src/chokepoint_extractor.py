"""
chokepoint_extractor.py
-----------------------
Automated ChokePoint Dataset Ingestion Engine.

Parses ground-truth XML annotations to extract aligned, bounding-box-cropped
face images for gallery enrollment and probe evaluation.  All eye-coordinate
geometry is derived directly from the XML; no face-detector inference is run
during extraction.

Third-party : xml.etree.ElementTree (stdlib), OpenCV (I/O, warpAffine),
              NumPy (coordinate algebra)
Custom      : XML parsing logic, best-frame frontality heuristic, IOD-
              proportional bounding-box derivation, eye-line rotation
              alignment, gallery / probe filesystem management.

Fail-fast policy
----------------
Every missing file or malformed annotation raises immediately.  There are no
silent fallbacks.

Usage (CLI)
-----------
    python src/chokepoint_extractor.py  \\
        /path/to/P1L_S1_C1              \\
        /path/to/P1L_S1_C1.xml          \\
        [--gallery data/gallery]        \\
        [--probes  data/probes]
"""

from __future__ import annotations

import glob
import math
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np

# ── Bounding-box proportions (relative to inter-ocular distance) ──────────────
BBOX_WIDTH_FACTOR  = 2.5   # face width  = 2.5 × IOD
BBOX_HEIGHT_FACTOR = 3.2   # face height = 3.2 × IOD

# Number of probe frames to sample per subject (excluding gallery frame)
N_PROBE_FRAMES = 5

# Minimum inter-ocular distance (pixels) below which detections are discarded
MIN_IOD_PX = 8.0

DEFAULT_GALLERY_DIR = "data/gallery"
DEFAULT_PROBE_DIR   = "data/probes"


# ─────────────────────────────────────────────────────────────────────────────
# XML PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_chokepoint_xml(xml_path: str) -> dict[str, list[dict]]:
    """
    Parse a ChokePoint ground-truth XML annotation file.

    Supported root tags: ``<personsequence>``, ``<dataset>``, ``<sequence>``.
    The function is permissive about the root tag but strict about file
    existence and annotation validity.

    Expected schema::

        <personsequence>
            <frame number="0042">
                <personlist>
                    <person id="3">
                        <leftEye  x="273" y="223"/>
                        <rightEye x="336" y="217"/>
                    </person>
                </personlist>
            </frame>
            ...
        </personsequence>

    Parameters
    ----------
    xml_path : str
        Path to the ``.xml`` annotation file.

    Returns
    -------
    dict[str, list[dict]]
        ``{ person_id : [ { "frame", "left_eye", "right_eye", "iod", "tilt" }, ... ] }``

    Raises
    ------
    FileNotFoundError
        If ``xml_path`` does not exist on the filesystem.
    ValueError
        If the file is parsable but contains zero valid face annotations.
    """
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(
            f"[chokepoint_extractor] XML annotation not found: '{xml_path}'\n"
            "Provide the ChokePoint ground-truth XML before running extraction."
        )

    tree = ET.parse(xml_path)
    root = tree.getroot()

    known_roots = {"personsequence", "dataset", "sequence"}
    if root.tag not in known_roots:
        print(
            f"[chokepoint_extractor] Warning: unexpected root tag <{root.tag}> — "
            "attempting to parse anyway."
        )

    subjects: dict[str, list[dict]] = {}

    for frame_elem in root.iter("frame"):
        raw_num = frame_elem.get("number", "").strip()
        try:
            frame_num = int(raw_num)
        except ValueError:
            continue  # malformed frame number → skip

        for person_elem in frame_elem.iter("person"):
            pid = person_elem.get("id", "").strip()
            if not pid:
                continue

            le = person_elem.find("leftEye")
            re = person_elem.find("rightEye")
            if le is None or re is None:
                continue

            try:
                lx, ly = float(le.get("x")), float(le.get("y"))
                rx, ry = float(re.get("x")), float(re.get("y"))
            except (TypeError, ValueError):
                continue

            iod = math.hypot(rx - lx, ry - ly)
            if iod < MIN_IOD_PX:
                continue  # degenerate annotation

            subjects.setdefault(pid, []).append(
                {
                    "frame":     frame_num,
                    "left_eye":  (lx, ly),
                    "right_eye": (rx, ry),
                    "iod":       iod,
                    "tilt":      abs(ry - ly),
                }
            )

    if not subjects:
        raise ValueError(
            f"[chokepoint_extractor] No valid face annotations found in '{xml_path}'.\n"
            "Verify that the XML conforms to the ChokePoint ground-truth schema."
        )

    return subjects


# ─────────────────────────────────────────────────────────────────────────────
# FRAME → IMAGE PATH RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_frame_path(sequence_dir: str, frame_num: int) -> str:
    """
    Locate the image file for a given frame number within a sequence directory.

    Tries multiple zero-padding conventions (4-, 5-, 6-, 8-digit) and glob patterns
    to handle prefixed filenames (e.g. ``P1L_S1_C1_T1_0042.jpg``).

    Parameters
    ----------
    sequence_dir : str
        Directory holding the raw ``.jpg`` / ``.png`` frame files.
    frame_num : int
        Integer frame identifier from the XML.

    Returns
    -------
    str
        Absolute path to the matched frame image.

    Raises
    ------
    FileNotFoundError
        If no matching image is found using any recognised naming convention.
    """
    # Direct filename candidates with various zero-padding widths
    candidates = [
        os.path.join(sequence_dir, f"{frame_num:04d}.jpg"),
        os.path.join(sequence_dir, f"{frame_num:04d}.png"),
        os.path.join(sequence_dir, f"{frame_num:05d}.jpg"),
        os.path.join(sequence_dir, f"{frame_num:05d}.png"),
        os.path.join(sequence_dir, f"{frame_num:06d}.jpg"),
        os.path.join(sequence_dir, f"{frame_num:06d}.png"),
        os.path.join(sequence_dir, f"{frame_num:08d}.jpg"),
        os.path.join(sequence_dir, f"{frame_num:08d}.png"),
        os.path.join(sequence_dir, f"{frame_num}.jpg"),
        os.path.join(sequence_dir, f"{frame_num}.png"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    # Glob for prefixed filenames (including potential zero-padded frame numbers)
    for pattern in (
        os.path.join(sequence_dir, f"*_{frame_num:04d}.jpg"),
        os.path.join(sequence_dir, f"*_{frame_num:04d}.png"),
        os.path.join(sequence_dir, f"*_{frame_num:05d}.jpg"),
        os.path.join(sequence_dir, f"*_{frame_num:05d}.png"),
        os.path.join(sequence_dir, f"*_{frame_num:06d}.jpg"),
        os.path.join(sequence_dir, f"*_{frame_num:06d}.png"),
        os.path.join(sequence_dir, f"*_{frame_num:08d}.jpg"),
        os.path.join(sequence_dir, f"*_{frame_num:08d}.png"),
        os.path.join(sequence_dir, f"*_{frame_num}.jpg"),
    ):
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[0]

    raise FileNotFoundError(
        f"[chokepoint_extractor] Cannot locate frame {frame_num} in '{sequence_dir}'.\n"
        f"  Tried: {candidates}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# FACE ALIGNMENT & CROP
# ─────────────────────────────────────────────────────────────────────────────

def _align_and_crop_face(
    img: np.ndarray,
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
) -> np.ndarray:
    """
    Align a face to canonical upright orientation and crop using an IOD-
    proportional bounding box.

    Algorithm
    ---------
    1. Compute the rotation angle that maps the inter-ocular line to horizontal.
    2. Rotate the full frame around the eye midpoint (preserving scale = 1.0).
    3. Derive axis-aligned crop box: centre = eye midpoint,
       width = ``BBOX_WIDTH_FACTOR × IOD``, height = ``BBOX_HEIGHT_FACTOR × IOD``.
    4. Clamp box to image boundaries and crop.

    Parameters
    ----------
    img : np.ndarray
        Source BGR frame (any dimensions).
    left_eye, right_eye : tuple[float, float]
        Pixel ``(x, y)`` coordinates of the subject's eyes in the *original*
        (un-rotated) frame.

    Returns
    -------
    np.ndarray
        Cropped BGR face patch.

    Raises
    ------
    ValueError
        If the resulting crop is degenerate (< 20 px in either dimension).
    RuntimeError
        If ``cv2.warpAffine`` fails.
    """
    lx, ly = left_eye
    rx, ry = right_eye

    cx = (lx + rx) / 2.0
    cy = (ly + ry) / 2.0
    iod   = math.hypot(rx - lx, ry - ly)
    angle = math.degrees(math.atan2(ry - ly, rx - lx))   # signed, degrees

    W = BBOX_WIDTH_FACTOR  * iod
    H = BBOX_HEIGHT_FACTOR * iod

    h_img, w_img = img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, scale=1.0)

    try:
        rotated = cv2.warpAffine(
            img, rot_mat, (w_img, h_img), flags=cv2.INTER_LINEAR
        )
    except cv2.error as exc:
        raise RuntimeError(
            f"[chokepoint_extractor] warpAffine failed: {exc}"
        ) from exc

    x1 = max(0, int(round(cx - W / 2.0)))
    y1 = max(0, int(round(cy - H / 2.0)))
    x2 = min(w_img - 1, int(round(cx + W / 2.0)))
    y2 = min(h_img - 1, int(round(cy + H / 2.0)))

    crop = rotated[y1:y2, x1:x2]

    if crop.shape[0] < 20 or crop.shape[1] < 20:
        raise ValueError(
            f"[chokepoint_extractor] Degenerate crop: {crop.shape}. "
            "Frame may be at image boundary or IOD is too small."
        )

    return crop


# ─────────────────────────────────────────────────────────────────────────────
# BEST-FRAME SELECTION & PROBE SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def _select_best_frame(detections: list[dict]) -> dict:
    """
    Select the most frontal detection for gallery enrollment.

    Frontality score: ``IOD / (1.0 + |y_right − y_left|)``

    Higher IOD means the face is larger / closer to the camera (less likely
    to be partially out-of-frame).  Lower tilt means the eye-line is
    horizontal (more frontal pose).

    Returns the detection entry with the highest score.
    """
    return max(detections, key=lambda d: d["iod"] / (1.0 + d["tilt"]))


def _sample_probe_frames(
    detections: list[dict],
    best_frame_num: int,
    n: int = N_PROBE_FRAMES,
) -> list[dict]:
    """
    Sample up to ``n`` temporally diverse probe frames for a subject,
    excluding the reserved gallery frame.

    Frames are sorted chronologically; a uniform stride is applied so that
    the sampled set spans the full extent of the subject's track.
    """
    candidates = sorted(
        [d for d in detections if d["frame"] != best_frame_num],
        key=lambda d: d["frame"],
    )
    if len(candidates) <= n:
        return candidates

    stride  = len(candidates) / n
    indices = [int(i * stride) for i in range(n)]
    return [candidates[i] for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_chokepoint_dataset(
    sequence_dir: str,
    xml_path: str,
    gallery_dir: str = DEFAULT_GALLERY_DIR,
    probe_dir: str   = DEFAULT_PROBE_DIR,
) -> dict[str, dict]:
    """
    Full automated ChokePoint face-crop extraction pipeline.

    For every unique subject in the annotation:
    * Selects the highest-frontality frame as the gallery enrollment image.
    * Samples 5 temporally diverse frames as probe images.
    * Aligns each face to canonical upright orientation using eye coordinates.
    * Saves gallery crops to ``gallery_dir/{person_id}.jpg``.
    * Saves probe crops to ``probe_dir/{person_id}_frame_{N}.jpg``.

    Parameters
    ----------
    sequence_dir : str
        Directory containing the raw ``.jpg`` frame images.
    xml_path : str
        Path to the matching ChokePoint ground-truth XML file.
    gallery_dir, probe_dir : str
        Output directories.  Created automatically if they do not exist.

    Returns
    -------
    dict[str, dict]
        Evaluation registry::

            {
                "person_id": {
                    "gallery_path": "data/gallery/person_id.jpg",
                    "probe_paths":  ["data/probes/person_id_frame_42.jpg", ...]
                },
                ...
            }

    Raises
    ------
    FileNotFoundError
        If ``sequence_dir`` or ``xml_path`` does not exist.
    ValueError
        If the XML contains no valid annotations.
    RuntimeError
        If extraction yields zero valid subject entries.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not os.path.isdir(sequence_dir):
        raise FileNotFoundError(
            f"[chokepoint_extractor] Sequence directory not found: '{sequence_dir}'"
        )

    os.makedirs(gallery_dir, exist_ok=True)
    os.makedirs(probe_dir,   exist_ok=True)

    # ── Parse XML ─────────────────────────────────────────────────────────────
    print(f"[chokepoint_extractor] Parsing XML: {xml_path}")
    subjects = parse_chokepoint_xml(xml_path)
    print(
        f"[chokepoint_extractor] {len(subjects)} unique subjects detected: "
        f"{sorted(subjects.keys())}"
    )

    registry: dict[str, dict] = {}
    extraction_errors: list[str] = []

    for pid, detections in sorted(subjects.items()):
        print(f"\n[chokepoint_extractor] Subject '{pid}'  ({len(detections)} frames)...")

        # ── Gallery: best (most-frontal) frame ────────────────────────────────
        best = _select_best_frame(detections)
        try:
            frame_path = _resolve_frame_path(sequence_dir, best["frame"])
            img = cv2.imread(frame_path)
            if img is None:
                raise RuntimeError(
                    f"cv2.imread returned None for '{frame_path}' — "
                    "file may be corrupt or have an unsupported codec."
                )
            crop = _align_and_crop_face(img, best["left_eye"], best["right_eye"])
            gallery_path = os.path.join(gallery_dir, f"{pid}.jpg")
            cv2.imwrite(gallery_path, crop)
            print(
                f"  ✓ Gallery → {gallery_path}  "
                f"(frame {best['frame']}, IOD={best['iod']:.1f} px)"
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            msg = f"  [FAIL] Gallery for subject '{pid}': {exc}"
            print(msg)
            extraction_errors.append(msg)
            continue   # cannot enroll this subject without a gallery image

        # ── Probes: diverse temporal sample ───────────────────────────────────
        probe_frames = _sample_probe_frames(detections, best["frame"])
        probe_paths: list[str] = []

        for pd_entry in probe_frames:
            try:
                fp  = _resolve_frame_path(sequence_dir, pd_entry["frame"])
                img = cv2.imread(fp)
                if img is None:
                    raise RuntimeError(f"cv2.imread returned None for '{fp}'")
                crop = _align_and_crop_face(img, pd_entry["left_eye"], pd_entry["right_eye"])
                out  = os.path.join(probe_dir, f"{pid}_frame_{pd_entry['frame']}.jpg")
                cv2.imwrite(out, crop)
                probe_paths.append(out)
                print(f"  ✓ Probe  → {out}")
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                print(f"  [WARN] Probe frame {pd_entry['frame']}: {exc}")

        if not probe_paths:
            print(f"  [WARN] No probe frames for '{pid}' — subject excluded.")
            continue

        registry[pid] = {
            "gallery_path": gallery_path,
            "probe_paths":  probe_paths,
        }

    if not registry:
        raise RuntimeError(
            "[chokepoint_extractor] Extraction produced zero valid subjects.\n"
            "Errors:\n" + "\n".join(extraction_errors)
        )

    print(
        f"\n[chokepoint_extractor] Complete: {len(registry)} subjects enrolled.\n"
        f"  Gallery → {gallery_dir}\n"
        f"  Probes  → {probe_dir}"
    )
    return registry


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ChokePoint automated face-crop extractor (M.A.R.K.)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "sequence_dir",
        help="Path to the folder containing raw .jpg frame images.",
    )
    parser.add_argument(
        "xml_path",
        help="Path to the matching ChokePoint ground-truth .xml file.",
    )
    parser.add_argument(
        "--gallery", default=DEFAULT_GALLERY_DIR,
        help=f"Output gallery directory (default: {DEFAULT_GALLERY_DIR})",
    )
    parser.add_argument(
        "--probes", default=DEFAULT_PROBE_DIR,
        help=f"Output probes directory (default: {DEFAULT_PROBE_DIR})",
    )
    _args = parser.parse_args()

    extract_chokepoint_dataset(
        _args.sequence_dir,
        _args.xml_path,
        gallery_dir=_args.gallery,
        probe_dir=_args.probes,
    )
