"""Integration tests for MediaPipe and the persisted classifier artifact."""

from __future__ import annotations

import mediapipe as mp
import numpy as np
import pytest

from face_emotion.constants import DEFAULT_EMOTION_MODEL
from face_emotion.features import build_feature_vector, compute_geometric_features, encode_blendshapes, landmarks_to_array, normalize_landmarks
from face_emotion.inference import EmotionClassifier
from face_emotion.landmarker import create_face_landmarker


def test_face_landmarker_returns_478_landmarks(landmarker_asset_path, sample_image_path) -> None:
    with create_face_landmarker(landmarker_asset_path, output_face_blendshapes=True) as detector:
        result = detector.detect(mp.Image.create_from_file(str(sample_image_path)))
    assert len(result.face_landmarks) == 1
    assert len(result.face_landmarks[0]) == 478
    assert len(result.face_blendshapes[0]) > 0


def test_feature_vector_shape_is_stable(landmarker_asset_path, sample_image_path) -> None:
    with create_face_landmarker(landmarker_asset_path, output_face_blendshapes=True) as detector:
        result = detector.detect(mp.Image.create_from_file(str(sample_image_path)))
    normalized = normalize_landmarks(landmarks_to_array(result.face_landmarks[0]))
    geometry = compute_geometric_features(normalized)
    blendshape_vector, names = encode_blendshapes(result.face_blendshapes[0])
    dense_geometry = build_feature_vector(normalized, geometry, blendshape_vector, "dense_geometry")
    geometry_blendshape = build_feature_vector(normalized, geometry, blendshape_vector, "geometry_blendshape")
    assert dense_geometry.shape == (1434 + 46,)
    assert geometry_blendshape.shape == (46 + len(names),)


def test_saved_model_probabilities_sum_to_one(sample_image_path) -> None:
    model_path = DEFAULT_EMOTION_MODEL
    if not model_path.exists():
        pytest.skip("O artefato treinado ainda nao foi gerado.")

    classifier = EmotionClassifier(model_path)
    with create_face_landmarker(classifier.metadata["landmarker_model"], output_face_blendshapes=True) as detector:
        result = detector.detect(mp.Image.create_from_file(str(sample_image_path)))

    probabilities = classifier.predict_probabilities(result.face_landmarks[0], result.face_blendshapes[0])
    assert probabilities.shape == (6,)
    assert np.isclose(float(probabilities.sum()), 1.0, atol=1e-5)
