import cv2
import numpy as np

from src.face_detector import FaceBox
from src.prepare_dataset import _process_image


class FakeDetector:
    def detect(self, _rgb):
        return [FaceBox(x=20, y=10, width=50, height=60)]


def test_process_image_preserves_224_square_output_with_tasks_detector(tmp_path):
    raw = tmp_path / "raw" / "Good"
    out = tmp_path / "processed" / "Good"
    raw.mkdir(parents=True)
    out.mkdir(parents=True)
    source = raw / "photo.jpg"
    cv2.imwrite(str(source), np.full((100, 100, 3), 120, dtype=np.uint8))

    count = _process_image(FakeDetector(), source, raw, out, margin=0, min_face_px=20)

    rendered = cv2.imread(str(out / "photo.jpg"))
    assert count == 1
    assert rendered.shape == (224, 224, 3)
