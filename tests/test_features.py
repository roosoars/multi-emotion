"""Unit tests for geometry, smoothing and threshold logic."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from face_emotion.features import build_feature_vector, compute_geometric_features, encode_blendshapes, landmarks_to_array, normalize_landmarks
from face_emotion.inference import EmotionClassifier
from face_emotion.smoothing import ProbabilitySmoother


class DummyEstimator:
    def predict_proba(self, _features: np.ndarray) -> np.ndarray:
        return np.array([[0.10, 0.08, 0.07, 0.61, 0.09, 0.05]], dtype=np.float32)


def test_normalize_landmarks_centers_on_reference(mock_landmarks) -> None:
    array = landmarks_to_array(mock_landmarks)
    normalized = normalize_landmarks(array)
    np.testing.assert_allclose(normalized[1], np.zeros(3, dtype=np.float32), atol=1e-7)


def test_compute_geometric_features_has_stable_shape(mock_landmarks) -> None:
    normalized = normalize_landmarks(landmarks_to_array(mock_landmarks))
    features = compute_geometric_features(normalized)
    assert features.shape == (46,)
    assert np.isfinite(features).all()


def test_build_feature_vectors_for_supported_feature_sets(mock_landmarks) -> None:
    normalized = normalize_landmarks(landmarks_to_array(mock_landmarks))
    geometry = compute_geometric_features(normalized)
    blend, names = encode_blendshapes([])
    dense_geometry = build_feature_vector(normalized, geometry, blend, "dense_geometry")
    geometry_blend = build_feature_vector(normalized, geometry, np.zeros(len(names), dtype=np.float32), "geometry_blendshape")
    assert dense_geometry.shape == (1434 + 46,)
    assert geometry_blend.shape == (46,)


def test_probability_smoother_average_and_reset() -> None:
    smoother = ProbabilitySmoother(window_size=3)
    first = smoother.update([0.7, 0.2, 0.1])
    second = smoother.update([0.4, 0.4, 0.2])
    np.testing.assert_allclose(first, np.array([0.7, 0.2, 0.1], dtype=np.float32))
    np.testing.assert_allclose(second, np.array([0.55, 0.3, 0.15], dtype=np.float32))
    smoother.reset()
    assert smoother.is_empty


def test_prediction_threshold_maps_to_indefinida(tmp_path: Path, mock_landmarks) -> None:
    artifact_path = tmp_path / "dummy.joblib"
    joblib.dump(
        {
            "model": DummyEstimator(),
            "labels": ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"],
            "feature_set": "dense_geometry",
            "blendshape_names": [],
            "confidence_threshold": 0.80,
        },
        artifact_path,
    )
    classifier = EmotionClassifier(artifact_path)
    prediction = classifier.predict_from_detection(mock_landmarks, [])
    assert prediction.raw_label == "Happiness"
    assert prediction.label == "Indefinida"
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-6
