"""Feature extraction helpers for face emotion classification."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .constants import (
    CLASS_NAMES,
    LEFT_EYEBROW,
    LEFT_EYE,
    LEFT_IRIS,
    LIPS_INNER,
    LIPS_OUTER,
    NOSE,
    RIGHT_EYEBROW,
    RIGHT_EYE,
    RIGHT_IRIS,
)

EPSILON = 1e-6

GEOMETRIC_FEATURE_NAMES = [
    "face_width",
    "face_height",
    "cheek_width",
    "eye_distance",
    "left_eye_width",
    "right_eye_width",
    "left_eye_opening",
    "right_eye_opening",
    "left_eye_aspect_ratio",
    "right_eye_aspect_ratio",
    "eye_opening_delta",
    "left_iris_offset_x",
    "right_iris_offset_x",
    "left_iris_offset_y",
    "right_iris_offset_y",
    "left_brow_eye_distance",
    "right_brow_eye_distance",
    "brow_distance_delta",
    "left_brow_width",
    "right_brow_width",
    "mouth_width",
    "mouth_opening",
    "mouth_aspect_ratio",
    "upper_lip_height",
    "lower_lip_height",
    "lip_height_ratio",
    "smile_left",
    "smile_right",
    "smile_delta",
    "nose_length",
    "nose_to_upper_lip",
    "nose_to_chin",
    "mouth_to_chin",
    "upper_face_height",
    "lower_face_height",
    "lower_upper_face_ratio",
    "left_jaw_to_eye",
    "right_jaw_to_eye",
    "left_eye_to_nose",
    "right_eye_to_nose",
    "left_mouth_to_cheek",
    "right_mouth_to_cheek",
    "inner_mouth_width",
    "outer_mouth_width",
    "jaw_angle_proxy",
    "lip_roundness",
]


def landmarks_to_array(landmarks: Iterable[object]) -> np.ndarray:
    """Convert MediaPipe landmarks to an `(N, 3)` float array."""
    points = np.array([[float(lm.x), float(lm.y), float(lm.z)] for lm in landmarks], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected landmarks to be convertible to an (N, 3) array.")
    return points


def normalize_landmarks(landmarks: np.ndarray, center_index: int = 1) -> np.ndarray:
    """Center and scale landmarks by the face bounding-box diagonal."""
    if landmarks.shape != (478, 3):
        raise ValueError(f"Expected 478 landmarks with xyz coordinates, got {landmarks.shape}.")
    centered = landmarks - landmarks[center_index]
    diagonal = np.linalg.norm(centered[:, :2].max(axis=0) - centered[:, :2].min(axis=0))
    if diagonal <= EPSILON:
        raise ValueError("Invalid landmark set with zero facial diagonal.")
    return centered / diagonal


def flatten_landmarks(normalized_landmarks: np.ndarray) -> np.ndarray:
    """Flatten normalized landmarks into a dense feature vector."""
    return normalized_landmarks.reshape(-1).astype(np.float32)


def _distance(points: np.ndarray, first: int, second: int) -> float:
    return float(np.linalg.norm(points[first, :2] - points[second, :2]))


def _midpoint(points: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return points[np.array(indices, dtype=np.int32)].mean(axis=0)


def compute_geometric_features(normalized_landmarks: np.ndarray) -> np.ndarray:
    """Build a stable set of geometry features from normalized landmarks."""
    points = normalized_landmarks

    face_width = _distance(points, 234, 454)
    face_height = _distance(points, 10, 152)
    cheek_width = _distance(points, 93, 323)
    eye_distance = _distance(points, 33, 263)

    left_eye_width = _distance(points, 33, 133)
    right_eye_width = _distance(points, 362, 263)
    left_eye_opening = _distance(points, 159, 145)
    right_eye_opening = _distance(points, 386, 374)
    left_eye_aspect = left_eye_opening / max(left_eye_width, EPSILON)
    right_eye_aspect = right_eye_opening / max(right_eye_width, EPSILON)

    left_iris_center = _midpoint(points, LEFT_IRIS)
    right_iris_center = _midpoint(points, RIGHT_IRIS)
    left_eye_center = _midpoint(points, [33, 133, 159, 145])
    right_eye_center = _midpoint(points, [362, 263, 386, 374])

    left_eye_half_width = max(left_eye_width / 2.0, EPSILON)
    right_eye_half_width = max(right_eye_width / 2.0, EPSILON)
    left_eye_half_height = max(left_eye_opening / 2.0, EPSILON)
    right_eye_half_height = max(right_eye_opening / 2.0, EPSILON)

    left_iris_offset_x = float((left_iris_center[0] - left_eye_center[0]) / left_eye_half_width)
    right_iris_offset_x = float((right_iris_center[0] - right_eye_center[0]) / right_eye_half_width)
    left_iris_offset_y = float((left_iris_center[1] - left_eye_center[1]) / left_eye_half_height)
    right_iris_offset_y = float((right_iris_center[1] - right_eye_center[1]) / right_eye_half_height)

    left_brow_eye_distance = _distance(points, 105, 159)
    right_brow_eye_distance = _distance(points, 334, 386)
    left_brow_width = _distance(points, LEFT_EYEBROW[0], LEFT_EYEBROW[-1])
    right_brow_width = _distance(points, RIGHT_EYEBROW[0], RIGHT_EYEBROW[-1])

    mouth_width = _distance(points, 61, 291)
    mouth_opening = _distance(points, 13, 14)
    mouth_aspect = mouth_opening / max(mouth_width, EPSILON)
    upper_lip_height = _distance(points, 0, 13)
    lower_lip_height = _distance(points, 14, 17)
    smile_left = _distance(points, 61, 1)
    smile_right = _distance(points, 291, 1)
    inner_mouth_width = _distance(points, 78, 308)
    outer_mouth_width = _distance(points, 61, 291)

    nose_length = _distance(points, NOSE[0], 2)
    nose_to_upper_lip = _distance(points, 1, 13)
    nose_to_chin = _distance(points, 1, 152)
    mouth_to_chin = _distance(points, 13, 152)
    upper_face_height = _distance(points, 10, 1)
    lower_face_height = _distance(points, 1, 152)

    left_jaw_to_eye = _distance(points, 234, 33)
    right_jaw_to_eye = _distance(points, 454, 263)
    left_eye_to_nose = _distance(points, 33, 1)
    right_eye_to_nose = _distance(points, 263, 1)
    left_mouth_to_cheek = _distance(points, 61, 234)
    right_mouth_to_cheek = _distance(points, 291, 454)

    jaw_angle_proxy = _distance(points, 58, 288) / max(face_width, EPSILON)
    lip_roundness = mouth_opening / max(inner_mouth_width, EPSILON)

    features = np.array(
        [
            face_width,
            face_height,
            cheek_width,
            eye_distance,
            left_eye_width,
            right_eye_width,
            left_eye_opening,
            right_eye_opening,
            left_eye_aspect,
            right_eye_aspect,
            abs(left_eye_aspect - right_eye_aspect),
            left_iris_offset_x,
            right_iris_offset_x,
            left_iris_offset_y,
            right_iris_offset_y,
            left_brow_eye_distance,
            right_brow_eye_distance,
            abs(left_brow_eye_distance - right_brow_eye_distance),
            left_brow_width,
            right_brow_width,
            mouth_width,
            mouth_opening,
            mouth_aspect,
            upper_lip_height,
            lower_lip_height,
            upper_lip_height / max(lower_lip_height, EPSILON),
            smile_left,
            smile_right,
            abs(smile_left - smile_right),
            nose_length,
            nose_to_upper_lip,
            nose_to_chin,
            mouth_to_chin,
            upper_face_height,
            lower_face_height,
            lower_face_height / max(upper_face_height, EPSILON),
            left_jaw_to_eye,
            right_jaw_to_eye,
            left_eye_to_nose,
            right_eye_to_nose,
            left_mouth_to_cheek,
            right_mouth_to_cheek,
            inner_mouth_width,
            outer_mouth_width,
            jaw_angle_proxy,
            lip_roundness,
        ],
        dtype=np.float32,
    )
    if features.shape[0] != len(GEOMETRIC_FEATURE_NAMES):
        raise AssertionError("Geometry feature count does not match feature names.")
    return features


def encode_blendshapes(
    blendshapes: Sequence[object],
    category_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert blendshapes into a stable numeric vector."""
    if category_names is None:
        category_names = [str(category.category_name) for category in blendshapes]
    name_to_score = {str(category.category_name): float(category.score or 0.0) for category in blendshapes}
    vector = np.array([name_to_score.get(name, 0.0) for name in category_names], dtype=np.float32)
    return vector, list(category_names)


def build_feature_vector(
    normalized_landmarks: np.ndarray,
    geometric_features: np.ndarray,
    blendshape_features: np.ndarray,
    feature_set: str,
) -> np.ndarray:
    """Assemble the feature vector required by a given candidate model."""
    if feature_set == "dense_geometry":
        return np.concatenate([flatten_landmarks(normalized_landmarks), geometric_features]).astype(np.float32)
    if feature_set == "geometry_blendshape":
        return np.concatenate([geometric_features, blendshape_features]).astype(np.float32)
    raise ValueError(f"Unsupported feature set: {feature_set}")


def labels_to_index() -> dict[str, int]:
    return {label: index for index, label in enumerate(CLASS_NAMES)}
