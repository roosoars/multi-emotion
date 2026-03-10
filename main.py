"""Realtime face emotion recognition using MediaPipe FaceLandmarker."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from mediapipe.tasks.python.vision import RunningMode

from face_emotion.constants import (
    DEFAULT_EMOTION_MODEL,
    DEFAULT_LANDMARKER_MODEL,
    SMOOTHING_WINDOW,
)
from face_emotion.inference import EmotionClassifier
from face_emotion.landmarker import LiveStreamLandmarker, bgr_frame_to_mp_image, create_face_landmarker
from face_emotion.smoothing import ProbabilitySmoother
from face_emotion.visualization import bounding_box, draw_bounding_box, draw_face_mesh, draw_prediction, landmarks_to_pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face emotion recognition with FaceLandmarker.")
    parser.add_argument("--mode", choices=("live", "video"), required=True, help="Modo de inferencia.")
    parser.add_argument("--input", type=Path, help="Arquivo de video para o modo video.")
    parser.add_argument("--camera-index", type=int, default=0, help="Indice da webcam no modo live.")
    parser.add_argument(
        "--show-mesh",
        action="store_true",
        help="Renderiza os 478 landmarks e os contornos mapeados.",
    )
    parser.add_argument(
        "--landmarker-model",
        type=Path,
        default=DEFAULT_LANDMARKER_MODEL,
        help="Caminho para o asset face_landmarker.task.",
    )
    parser.add_argument(
        "--emotion-model",
        type=Path,
        default=DEFAULT_EMOTION_MODEL,
        help="Caminho para o classificador de emocoes treinado.",
    )
    args = parser.parse_args()
    if args.mode == "video" and args.input is None:
        parser.error("--input e obrigatorio quando --mode video.")
    return args


def _render_result(
    frame,
    classifier: EmotionClassifier,
    smoother: ProbabilitySmoother,
    result: object | None,
    *,
    show_mesh: bool,
    status: str,
) -> None:
    if result is None or not result.face_landmarks:
        smoother.reset()
        draw_prediction(frame, "Indefinida", 0.0, status="Sem rosto detectado")
        return

    face_landmarks = result.face_landmarks[0]
    blendshapes = result.face_blendshapes[0] if result.face_blendshapes else []
    probabilities = classifier.predict_probabilities(face_landmarks, blendshapes)
    smoothed = smoother.update(probabilities)
    prediction = classifier.predict_from_detection(face_landmarks, blendshapes, probabilities_override=smoothed)

    if show_mesh:
        draw_face_mesh(frame, face_landmarks, draw_points=True)
    else:
        points = landmarks_to_pixels(face_landmarks, frame.shape[1], frame.shape[0])
        draw_bounding_box(frame, bounding_box(points))

    draw_prediction(frame, prediction.label, prediction.confidence, status=status)


def run_live(args: argparse.Namespace) -> None:
    classifier = EmotionClassifier(args.emotion_model)
    smoother = ProbabilitySmoother(SMOOTHING_WINDOW)
    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a webcam no indice {args.camera_index}.")

    tracker = LiveStreamLandmarker(args.landmarker_model, output_face_blendshapes=True)
    try:
        while camera.isOpened():
            ok, frame = camera.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            timestamp_ms = int(time.monotonic() * 1000)
            tracker.submit(frame, timestamp_ms)
            result, result_timestamp = tracker.latest()
            if result_timestamp >= 0 and timestamp_ms - result_timestamp > 400:
                result = None
            status = f"Modo live | timestamp={result_timestamp}" if result_timestamp >= 0 else "Modo live | aguardando callback"
            _render_result(
                frame,
                classifier,
                smoother,
                result,
                show_mesh=args.show_mesh,
                status=status,
            )
            cv2.imshow("Face Emotion - Live", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


def run_video(args: argparse.Namespace) -> None:
    classifier = EmotionClassifier(args.emotion_model)
    smoother = ProbabilitySmoother(SMOOTHING_WINDOW)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video {args.input}.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    with create_face_landmarker(
        args.landmarker_model,
        running_mode=RunningMode.VIDEO,
        output_face_blendshapes=True,
    ) as landmarker:
        frame_index = 0
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0:
                timestamp_ms = int((frame_index / fps) * 1000)
            result = landmarker.detect_for_video(
                image=bgr_frame_to_mp_image(frame),
                timestamp_ms=timestamp_ms,
            )
            status = f"Modo video | frame={frame_index}"
            _render_result(
                frame,
                classifier,
                smoother,
                result,
                show_mesh=args.show_mesh,
                status=status,
            )
            cv2.imshow("Face Emotion - Video", frame)
            frame_index += 1
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.mode == "live":
        run_live(args)
        return
    run_video(args)


if __name__ == "__main__":
    main()
