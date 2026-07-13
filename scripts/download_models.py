"""Download the official MediaPipe face model with integrity verification."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
MODEL_SHA256 = "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "blaze_face_short_range.tflite"


def verify_model(path: Path, expected_sha256: str = MODEL_SHA256) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError("model checksum mismatch; refusing to use downloaded file")


def download_model(destination: Path = MODEL_PATH) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        verify_model(destination)
        return destination
    fd, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".download")
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        # Fixed HTTPS origin plus SHA-256 verification below prevents untrusted assets.
        urllib.request.urlretrieve(MODEL_URL, temporary)  # nosec B310
        verify_model(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


if __name__ == "__main__":
    print(download_model())
