"""
人臉裁切前處理 — 把原始照片轉成 224x224 的人臉訓練樣本。

輸入結構（raw 資料夾）：
    datasets/raw/Good/*.jpg
    datasets/raw/Bad/*.jpg

輸出結構（processed 資料夾）：
    datasets/processed/Good/*.jpg
    datasets/processed/Bad/*.jpg

流程：
    1. MediaPipe FaceDetection 找出每張照片的人臉 bounding box
    2. 向外擴張 margin（預設 15%）保留頭部與少量背景
    3. resize 成 224x224
    4. 一張照片若有多張臉，每張各自存一個檔
    5. 沒偵測到臉的照片會被跳過並紀錄
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp

CLASSES = ("Good", "Bad")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_SIZE = 224


def _expand_box(
    x: int, y: int, w: int, h: int,
    img_w: int, img_h: int, margin: float,
) -> tuple[int, int, int, int]:
    """以中心點為基準向外擴張 margin 比例，並裁切到影像邊界內。"""
    cx, cy = x + w / 2, y + h / 2
    new_w = w * (1 + margin)
    new_h = h * (1 + margin)
    side = max(new_w, new_h)
    x1 = int(max(0, cx - side / 2))
    y1 = int(max(0, cy - side / 2))
    x2 = int(min(img_w, cx + side / 2))
    y2 = int(min(img_h, cy + side / 2))
    return x1, y1, x2, y2


def _safe_stem(img_path: Path, raw_class_dir: Path) -> str:
    """把相對路徑的所有父資料夾 + 檔名串成 'folderA__sub__photo' 形式。

    避免不同子資料夾下同名照片(IMG_001.jpg 之類)互相覆蓋。
    若檔案直接在 raw_class_dir 底下、沒有子資料夾,就直接用原檔名。
    """
    try:
        rel = img_path.relative_to(raw_class_dir)
    except ValueError:
        return img_path.stem
    parts = list(rel.parts)
    if len(parts) == 1:  # 散裝在 raw/Good/ 或 raw/Bad/ 底下
        return img_path.stem
    # 把空白與不安全字元換成底線,避免檔名怪
    safe_parts = [p.replace(" ", "_").replace("/", "_") for p in parts[:-1]]
    return "__".join(safe_parts) + "__" + img_path.stem


def _crop_letterbox_bgr(bgr, std_threshold: float = 8.0):
    """裁掉純色邊框(處理 iPhone 截圖、影片黑邊等)。回傳裁切後影像 + offset。"""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    max_cv, max_ch = h // 4, w // 4
    top = bottom = left = right = 0
    for i in range(max_cv):
        if gray[i, :].std() > std_threshold:
            top = i; break
    for i in range(max_cv):
        if gray[h - 1 - i, :].std() > std_threshold:
            bottom = i; break
    for i in range(max_ch):
        if gray[:, i].std() > std_threshold:
            left = i; break
    for i in range(max_ch):
        if gray[:, w - 1 - i].std() > std_threshold:
            right = i; break
    if top + bottom + left + right == 0:
        return bgr, (0, 0)
    return bgr[top:h - bottom, left:w - right], (left, top)


def _process_image(
    detector,
    img_path: Path,
    raw_class_dir: Path,
    out_dir: Path,
    margin: float,
    min_face_px: int,
    fallback_detector=None,
) -> int:
    """處理單張照片,回傳實際輸出張數。

    三層 fallback:
        1. primary detector(model 0)
        2. fallback detector(model 1)
        3. 裁掉 letterbox 黑邊後再跑 1+2
    輸出檔名會加上父資料夾前綴,避免重名衝突。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    if (not result.detections) and fallback_detector is not None:
        result = fallback_detector.process(rgb)

    # 第 3 層 fallback:裁掉純色邊框再試
    work_img = img
    offset_x, offset_y = 0, 0
    if not result.detections:
        cropped, (offset_x, offset_y) = _crop_letterbox_bgr(img)
        if cropped.shape != img.shape:
            rgb_c = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb_c)
            if (not result.detections) and fallback_detector is not None:
                result = fallback_detector.process(rgb_c)
            if result.detections:
                work_img = cropped

    if not result.detections:
        return 0

    out_stem = _safe_stem(img_path, raw_class_dir)

    # bbox 是相對於 work_img(可能是裁過 letterbox 後的);要加上 offset 才能落回原圖
    h_work, w_work = work_img.shape[:2]

    saved = 0
    for i, det in enumerate(result.detections):
        bbox = det.location_data.relative_bounding_box
        fx = int(bbox.xmin * w_work) + offset_x
        fy = int(bbox.ymin * h_work) + offset_y
        fw = int(bbox.width * w_work)
        fh = int(bbox.height * h_work)
        if fw < min_face_px or fh < min_face_px:
            continue
        # _expand_box 用原圖大小做邊界 clip
        x1, y1, x2, y2 = _expand_box(fx, fy, fw, fh, w, h, margin)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]
        face = cv2.resize(crop, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        suffix = "" if len(result.detections) == 1 else f"_face{i}"
        out_path = out_dir / f"{out_stem}{suffix}.jpg"
        cv2.imwrite(str(out_path), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    return saved


def prepare_class(
    detector,
    raw_class_dir: Path,
    out_class_dir: Path,
    margin: float,
    min_face_px: int,
    fallback_detector=None,
) -> tuple[int, int, list[Path]]:
    """處理單一類別資料夾，回傳 (input_count, output_count, skipped_paths)。"""
    out_class_dir.mkdir(parents=True, exist_ok=True)
    # rglob: 遞迴掃所有子資料夾,讓使用者可以直接把整包下載的資料夾丟進來
    images = sorted(
        p for p in raw_class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    skipped: list[Path] = []
    out_count = 0
    for p in images:
        n = _process_image(
            detector, p, raw_class_dir, out_class_dir,
            margin, min_face_px, fallback_detector,
        )
        if n == 0:
            skipped.append(p)
        out_count += n
    return len(images), out_count, skipped


def prepare_dataset(
    raw_root: Path,
    out_root: Path,
    margin: float = 0.15,
    min_face_px: int = 40,
    min_confidence: float = 0.5,
) -> None:
    if not raw_root.is_dir():
        raise FileNotFoundError(f"找不到原始資料夾：{raw_root}")
    for cls in CLASSES:
        if not (raw_root / cls).is_dir():
            raise FileNotFoundError(
                f"原始資料夾缺少 '{cls}' 子資料夾：預期 {raw_root / cls}"
            )

    out_root.mkdir(parents=True, exist_ok=True)
    # 雙模型 fallback 策略:
    # primary  = model 0 (短距離 / 大臉,擅長肖像、自拍、頭像)
    # fallback = model 1 (長距離 / 小臉,擅長合照中的中遠距人物)
    # 先試 primary,沒抓到再試 fallback,兩個的優點都吃到
    primary = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=min_confidence,
    )
    fallback = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=min_confidence,
    )

    print(f"[info] raw → {raw_root}")
    print(f"[info] out → {out_root}")
    print(f"[info] margin={margin:.0%}, min_face_px={min_face_px}, "
          f"min_confidence={min_confidence}, dual-model fallback ON")

    try:
        for cls in CLASSES:
            print(f"\n[{cls}] 處理中…")
            n_in, n_out, skipped = prepare_class(
                primary,
                raw_root / cls,
                out_root / cls,
                margin,
                min_face_px,
                fallback_detector=fallback,
            )
            print(f"  輸入 {n_in} 張 → 輸出 {n_out} 張人臉")
            if skipped:
                print(f"  ⚠️  {len(skipped)} 張未偵測到合格人臉（前 5 個）：")
                for p in skipped[:5]:
                    print(f"    - {p.name}")
    finally:
        primary.close()
        fallback.close()

    print("\n[done] 前處理完成。")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MediaPipe 人臉裁切：把原始照片轉成 224x224 訓練樣本。",
    )
    p.add_argument(
        "--raw", type=Path, default=Path("datasets/raw"),
        help="原始資料夾，內含 Good/ 與 Bad/（預設 datasets/raw）",
    )
    p.add_argument(
        "--out", type=Path, default=Path("datasets/processed"),
        help="輸出資料夾，會建立 Good/ 與 Bad/（預設 datasets/processed）",
    )
    p.add_argument(
        "--margin", type=float, default=0.15,
        help="人臉框向外擴張比例（預設 0.15 = 15%%）",
    )
    p.add_argument(
        "--min-face-px", type=int, default=40,
        help="人臉最短邊像素門檻，過小的偵測結果忽略（預設 40）",
    )
    p.add_argument(
        "--min-confidence", type=float, default=0.5,
        help="MediaPipe 信心門檻（預設 0.5）",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        prepare_dataset(
            args.raw, args.out,
            margin=args.margin,
            min_face_px=args.min_face_px,
            min_confidence=args.min_confidence,
        )
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
