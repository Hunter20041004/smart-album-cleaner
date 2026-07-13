"""
表情分類推論 — 給一張照片，回傳 Good/Bad 與信心。

設計重點：
    - 對外 API: predict_image_quality(image_path_or_array, model_path)
    - 模型 / MediaPipe detector 用模組級快取，重複呼叫不會反覆載入
    - 人臉裁切邏輯與 prepare_dataset.py 共用 _expand_box，
      確保訓練 / 推論的 crop 行為一致（避免 train-serve skew）
    - 沒偵測到人臉時拋出 NoFaceDetectedError，UI 端 try/except 即可
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.face_detector import FaceDetector
from src.prepare_dataset import _expand_box
from src.train_mobilenet import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    build_model,
)

DEFAULT_MODEL_PATH = Path("models/mobilenet_face.pth")
ImageInput = str | Path | np.ndarray


class NoFaceDetectedError(Exception):
    """整張照片都偵測不到合格人臉時拋出。UI 端可據此跳過該張。"""


# ── 模組級快取 ───────────────────────────────────────────────────────────────
_model_cache: dict[str, tuple[torch.nn.Module, list[str], torch.device, float]] = {}
_detector: FaceDetector | None = None
_detector_fallback = None  # 保留名稱，讓舊呼叫端清快取時不會壞掉
_eval_tf: transforms.Compose | None = None


def _get_detector() -> FaceDetector:
    """Lazy-load the supported MediaPipe Tasks detector."""
    global _detector
    if _detector is None:
        _detector = FaceDetector(min_confidence=0.3)
    return _detector


def _get_eval_transform() -> transforms.Compose:
    global _eval_tf
    if _eval_tf is None:
        _eval_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return _eval_tf


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(
    model_path: Path,
) -> tuple[torch.nn.Module, list[str], torch.device, float]:
    key = str(model_path.resolve())
    if key in _model_cache:
        return _model_cache[key]
    if not model_path.is_file():
        raise FileNotFoundError(
            f"找不到模型檔:{model_path}(請先跑 train_mobilenet.py)"
        )
    device = _pick_device()
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    classes = ckpt.get("classes")
    if not classes:
        raise RuntimeError(
            f"模型檔 {model_path} 缺少 'classes' 欄位 — 請用最新版 train_mobilenet.py 重新訓練"
        )
    # 訓練時掃出來的最佳 P(Bad) 閾值；舊 checkpoint 沒有就退回 0.5
    decision_threshold = float(ckpt.get("decision_threshold", 0.5))
    # 舊 checkpoint 沒寫 arch,預設 v2 (向後相容)
    arch = ckpt.get("arch", "mobilenet_v2")
    model = build_model(arch).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _model_cache[key] = (model, classes, device, decision_threshold)
    return _model_cache[key]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_bgr(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"輸入 ndarray 形狀必須為 (H, W, 3)，實際 {image.shape}")
        return image
    p = Path(image)
    if not p.is_file():
        raise FileNotFoundError(f"找不到照片：{p}")
    bgr = cv2.imread(str(p))
    if bgr is None:
        raise ValueError(f"無法讀取照片（格式不支援或損毀）：{p}")
    return bgr


def _crop_letterbox(bgr: np.ndarray, std_threshold: float = 8.0) -> tuple[np.ndarray, tuple[int, int]]:
    """偵測並裁掉純色 letterbox 邊框(iPhone 螢幕截圖的黑邊、影片黑邊等)。

    回傳 (裁切後影像, (offset_x, offset_y)) — offset 讓 bbox 能換算回原圖座標。
    最多裁掉每邊 1/4。
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    max_crop_v = h // 4
    max_crop_h = w // 4

    top = bottom = left = right = 0
    for i in range(max_crop_v):
        if gray[i, :].std() > std_threshold:
            top = i
            break
    for i in range(max_crop_v):
        if gray[h - 1 - i, :].std() > std_threshold:
            bottom = i
            break
    for i in range(max_crop_h):
        if gray[:, i].std() > std_threshold:
            left = i
            break
    for i in range(max_crop_h):
        if gray[:, w - 1 - i].std() > std_threshold:
            right = i
            break

    if top + bottom + left + right == 0:
        return bgr, (0, 0)  # 沒 letterbox,原樣回傳
    return bgr[top:h - bottom, left:w - right], (left, top)


def _detect_largest_face(
    bgr: np.ndarray, min_face_px: int, margin: float,
) -> tuple[int, int, int, int]:
    """回傳 (x1, y1, x2, y2);找不到合格人臉則丟 NoFaceDetectedError。

    先以 MediaPipe Tasks 偵測原圖；若失敗，裁掉 letterbox 邊框後重試。
    """
    detector = _get_detector()

    def _try_detect(img_bgr: np.ndarray):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return detector.detect(rgb)

    # 第 1 層：原圖直接偵測
    result = _try_detect(bgr)
    offset_x, offset_y = 0, 0
    if not result:
        # 第 2 層：裁 letterbox 後再試
        cropped, (offset_x, offset_y) = _crop_letterbox(bgr)
        if cropped.shape != bgr.shape:  # 真的裁了東西才重試
            result = _try_detect(cropped)
    if not result:
        raise NoFaceDetectedError(
            "MediaPipe Tasks（含 letterbox 裁切）找不到人臉"
        )

    H, W = bgr.shape[:2]  # 原圖大小 — 最後要回傳的座標是相對於原圖的

    best = None
    best_area = 0
    for bbox in result:
        fw = bbox.width
        fh = bbox.height
        if fw < min_face_px or fh < min_face_px:
            continue
        area = fw * fh
        if area > best_area:
            best_area = area
            # 換算回原圖座標(加上 letterbox offset)
            fx = bbox.x + offset_x
            fy = bbox.y + offset_y
            best = (fx, fy, fw, fh)
    if best is None:
        raise NoFaceDetectedError(
            f"偵測到的人臉皆小於 {min_face_px}px 門檻 — 跳過"
        )
    fx, fy, fw, fh = best
    return _expand_box(fx, fy, fw, fh, W, H, margin)


# ── Public API ───────────────────────────────────────────────────────────────
def predict_image_quality(
    image_path_or_array: ImageInput,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    margin: float = 0.15,
    min_face_px: int = 20,
) -> dict:
    """
    對單張照片做表情品質推論。

    Args:
        image_path_or_array: 照片路徑（str/Path）或 BGR ndarray (H, W, 3)
        model_path: 訓練輸出的 .pth 檔
        margin: 人臉框向外擴張比例（與 prepare_dataset 一致，預設 15%）
        min_face_px: 人臉最短邊像素門檻

    Returns:
        {
            "label":       "Good" 或 "Bad",
            "probability": float,           # 預測類別的機率 0~1
            "probs":       {"Good": ..., "Bad": ...},  # 兩類完整機率
            "face_image":  np.ndarray,      # 224x224 RGB 裁切結果（給 UI 顯示）
        }

    Raises:
        NoFaceDetectedError: 偵測不到合格人臉
        FileNotFoundError:   照片或模型檔不存在
        ValueError:          照片格式或形狀不正確
    """
    model, classes, device, decision_threshold = _load_model(Path(model_path))

    bgr = _load_bgr(image_path_or_array)
    x1, y1, x2, y2 = _detect_largest_face(bgr, min_face_px, margin)
    crop_bgr = bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        raise NoFaceDetectedError("人臉框經邊界裁切後為空 — 跳過")

    face_224_bgr = cv2.resize(
        crop_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA
    )
    face_224_rgb = cv2.cvtColor(face_224_bgr, cv2.COLOR_BGR2RGB)

    # ── TTA: 原圖 + 水平翻轉 兩個版本一起餵,平均機率 ─────────────
    tf = _get_eval_transform()
    face_flipped_rgb = face_224_rgb[:, ::-1, :].copy()  # 水平翻轉
    batch = torch.stack([
        tf(face_224_rgb),
        tf(face_flipped_rgb),
    ], dim=0).to(device)
    with torch.no_grad():
        probs_batch = F.softmax(model(batch), dim=1).cpu().numpy()
    probs = probs_batch.mean(axis=0)  # 兩版本平均 → 更穩定的機率

    # ── 用訓練時調好的閾值決定 label,而非死板的 argmax ─────────────
    bad_idx = classes.index("Bad")
    good_idx = classes.index("Good")
    p_bad = float(probs[bad_idx])
    if p_bad >= decision_threshold:
        pred_idx = bad_idx
    else:
        pred_idx = good_idx

    return {
        "label": classes[pred_idx],
        "probability": float(probs[pred_idx]),
        "probs": {cls: float(probs[i]) for i, cls in enumerate(classes)},
        "face_image": face_224_rgb,
        "decision_threshold": decision_threshold,  # 給 UI 顯示用
    }


# ── CLI: 給開發時手動驗證一張照片用 ─────────────────────────────────────────
def _cli() -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="對單張照片做 Good/Bad 表情推論。")
    p.add_argument("image", type=Path, help="照片路徑")
    p.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH,
        help=f"模型 .pth 路徑（預設 {DEFAULT_MODEL_PATH}）",
    )
    args = p.parse_args()

    try:
        out = predict_image_quality(args.image, args.model)
    except NoFaceDetectedError as e:
        print(f"[skip] {args.image}: {e}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    print(f"label       : {out['label']}")
    print(f"probability : {out['probability']:.4f}")
    print(f"probs       : {out['probs']}")
    print(f"face_image  : shape={out['face_image'].shape} (RGB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
