"""
影片轉影格工具 — 為眨眼偵測訓練資料集而設計。

從一支影片中每隔 N 幀擷取一張 JPG，依序流水號命名輸出到指定資料夾。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    output_dir: Path,
    frame_skip: int = 3,
    jpeg_quality: int = 95,
) -> int:
    """從 video_path 擷取影格到 output_dir，回傳實際儲存張數。"""
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到影片檔：{video_path}")
    if frame_skip < 1:
        raise ValueError(f"frame_skip 必須 >= 1（目前 {frame_skip}）")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 無法開啟影片：{video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    stem = video_path.stem
    print(f"[info] 影片: {video_path.name}  總幀數≈{total}  FPS≈{fps:.2f}")
    print(f"[info] 每 {frame_skip} 幀擷取 1 張 → {output_dir}")

    saved = 0
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_skip == 0:
                out_path = output_dir / f"{stem}_{saved:06d}.jpg"
                cv2.imwrite(
                    str(out_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                saved += 1
                if saved % 50 == 0:
                    print(f"  ...已儲存 {saved} 張")
            frame_idx += 1
    finally:
        cap.release()

    print(f"[done] 共讀取 {frame_idx} 幀，儲存 {saved} 張到 {output_dir}")
    return saved


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="從影片擷取影格，作為眨眼偵測訓練資料集。",
    )
    p.add_argument("video", type=Path, help="輸入影片路徑（mp4/mov/...）")
    p.add_argument("output", type=Path, help="輸出資料夾路徑")
    p.add_argument(
        "-s", "--frame-skip", type=int, default=3,
        help="每 N 幀擷取一張（預設 3，避免相鄰幀過於重複）",
    )
    p.add_argument(
        "-q", "--quality", type=int, default=95,
        help="JPEG 品質 0-100（預設 95）",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        extract_frames(args.video, args.output, args.frame_skip, args.quality)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
