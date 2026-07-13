"""Small MediaPipe Tasks adapter used by training and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np

DEFAULT_FACE_MODEL = Path("models/blaze_face_short_range.tflite")


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    width: int
    height: int


def boxes_from_result(result: Any) -> list[FaceBox]:
    return [
        FaceBox(
            x=int(detection.bounding_box.origin_x),
            y=int(detection.bounding_box.origin_y),
            width=int(detection.bounding_box.width),
            height=int(detection.bounding_box.height),
        )
        for detection in (result.detections or [])
    ]


class FaceDetector:
    """Absolute-pixel face boxes backed by the supported Tasks API."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_FACE_MODEL,
        min_confidence: float = 0.3,
    ) -> None:
        model = Path(model_path)
        if not model.is_file():
            raise FileNotFoundError(
                f"找不到人臉偵測模型：{model}；請執行 python scripts/download_models.py"
            )
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def detect(self, rgb: np.ndarray) -> list[FaceBox]:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return boxes_from_result(self._detector.detect(image))

    def close(self) -> None:
        self._detector.close()
