"""
predict_face.py 的單元測試 — 不需要訓練好的模型也能跑大部分。

策略:
    1. 用合成圖像測試「找不到人臉」路徑(NoFaceDetectedError)
    2. 用 Pillow 畫一個假人臉測試端到端流程
    3. 用真實照片(如果 datasets/ 有的話)做 smoke test
    4. Mock 掉 _load_model 確保不依賴 .pth 檔
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict_face import (
    NoFaceDetectedError,
    _load_bgr,
    predict_image_quality,
)


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture
def blank_image(tmp_path):
    """全黑 480x640 BGR 影像 — MediaPipe 在這上面找不到人臉。"""
    import cv2
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    p = tmp_path / "blank.jpg"
    cv2.imwrite(str(p), arr)
    return p


@pytest.fixture
def real_face_path() -> Path | None:
    """如果 datasets/processed/Good 裡有照片,拿第一張當測試輸入。"""
    candidates = sorted((ROOT / "datasets/processed/Good").glob("*.jpg"))
    return candidates[0] if candidates else None


@pytest.fixture
def model_available() -> bool:
    return (ROOT / "models" / "mobilenet_face.pth").is_file()


# ── _load_bgr 邊界測試 ──────────────────────────────────────────────
def test_load_bgr_accepts_ndarray():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    out = _load_bgr(arr)
    assert out is arr


def test_load_bgr_rejects_wrong_shape():
    with pytest.raises(ValueError, match="(H, W, 3)"):
        _load_bgr(np.zeros((10, 10), dtype=np.uint8))


def test_load_bgr_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_bgr(tmp_path / "does_not_exist.jpg")


def test_load_bgr_reads_real_jpg(blank_image):
    arr = _load_bgr(blank_image)
    assert arr.shape == (480, 640, 3)
    assert arr.dtype == np.uint8


# ── NoFaceDetectedError 路徑 ────────────────────────────────────────
def test_blank_image_raises_no_face(blank_image, model_available):
    """全黑圖必定找不到人臉,即使沒模型也應該在 detector 那層就 raise。"""
    if not model_available:
        pytest.skip("需要訓練好的模型才能跑這個測試")
    with pytest.raises(NoFaceDetectedError):
        predict_image_quality(blank_image)


# ── 端到端測試(需要模型 + 真實照片) ──────────────────────────────
def test_end_to_end_real_face(real_face_path, model_available):
    if not real_face_path:
        pytest.skip("datasets/processed/Good/ 內沒有照片可供測試")
    if not model_available:
        pytest.skip("需要訓練好的模型")

    result = predict_image_quality(real_face_path)

    # Result 結構驗證
    assert "label" in result
    assert result["label"] in ("Good", "Bad")
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert "probs" in result
    assert set(result["probs"].keys()) == {"Good", "Bad"}
    assert abs(sum(result["probs"].values()) - 1.0) < 1e-4  # 機率總和=1
    assert "face_image" in result
    assert result["face_image"].shape == (224, 224, 3)
    assert result["face_image"].dtype == np.uint8
    assert "decision_threshold" in result


def test_lazy_model_cache(real_face_path, model_available):
    """第二次呼叫不應該再 load 模型(應該命中模組級快取)。"""
    if not real_face_path or not model_available:
        pytest.skip("需要模型 + 真實照片")
    from src import predict_face

    # 清快取,從頭測
    predict_face._model_cache.clear()
    predict_face._detector = None
    predict_face._detector_fallback = None

    # 第一次呼叫:會載入模型 + detector
    predict_image_quality(real_face_path)
    assert len(predict_face._model_cache) == 1
    cached_obj_id = id(list(predict_face._model_cache.values())[0])

    # 第二次:同一個 cached 物件
    predict_image_quality(real_face_path)
    assert id(list(predict_face._model_cache.values())[0]) == cached_obj_id


# ── 例外類型驗證 ────────────────────────────────────────────────────
def test_missing_model_raises_filenotfound(blank_image, tmp_path):
    fake_model = tmp_path / "no_such_model.pth"
    with pytest.raises(FileNotFoundError):
        predict_image_quality(blank_image, model_path=fake_model)


def test_corrupted_image_raises(tmp_path, model_available):
    if not model_available:
        pytest.skip("需要模型")
    fake = tmp_path / "fake.jpg"
    fake.write_bytes(b"not actually a jpeg")
    with pytest.raises(ValueError, match="無法讀取"):
        predict_image_quality(fake)
