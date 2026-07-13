import hashlib

import pytest

from scripts.download_models import MODEL_SHA256, verify_model


def test_verified_model_accepts_expected_sha256(tmp_path):
    model = tmp_path / "model.tflite"
    model.write_bytes(b"verified model")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()
    verify_model(model, expected)


def test_verified_model_rejects_checksum_mismatch(tmp_path):
    model = tmp_path / "model.tflite"
    model.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum"):
        verify_model(model, MODEL_SHA256)
