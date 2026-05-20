"""
FastAPI 後端 — AI 表情相簿管家 v2.0

啟動: uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

# 引入既有的推論模組
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict_face import NoFaceDetectedError, predict_image_quality  # noqa: E402

# ──────────────────────────────────────────────────────────────────────────
# 常數
# ──────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "cache"
FACE_CACHE = CACHE_DIR / "faces"
FACE_CACHE.mkdir(parents=True, exist_ok=True)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_face.pth"

# Frontend build output(production 才用)
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"

# ──────────────────────────────────────────────────────────────────────────
# 全局狀態:scan jobs
# ──────────────────────────────────────────────────────────────────────────
JOBS: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Darkroom · AI Photo Curator API",
    version="2.0.0",
    description="FastAPI 後端,提供照片掃描、Trash 管理、影像縮圖等服務。",
)

# CORS — dev 時 Vue 在 5173,production 同源就不需要
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    folder: str


class TrashRequest(BaseModel):
    folder: str  # 掃描的根資料夾
    paths: list[str]  # 要刪除的絕對路徑


class RestoreRequest(BaseModel):
    folder: str
    trash_paths: list[str] | None = None  # None = 全部還原


# ──────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.is_file(),
        "model_path": str(MODEL_PATH),
    }


# ──────────────────────────────────────────────────────────────────────────
# 掃描相關
# ──────────────────────────────────────────────────────────────────────────
def _list_scannable_images(folder: Path) -> list[Path]:
    """遞迴列出可掃描的圖片,排除 Trash 子資料夾。"""
    out: list[Path] = []
    for f in folder.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in VALID_EXTS:
            continue
        try:
            rel_parts = f.relative_to(folder).parts
            if rel_parts and rel_parts[0] == "Trash":
                continue
        except ValueError:
            pass
        out.append(f)
    return sorted(out)


def _save_face_thumb(face_rgb, item_path: Path) -> str:
    """把 AI 裁切的人臉(numpy RGB)存成 JPEG,回傳 cache 檔名(供前端取用)。"""
    # 用原始檔路徑 + mtime 當 hash key,確保檔案不變就直接用 cache
    key = hashlib.md5(
        f"{item_path}::{item_path.stat().st_mtime}".encode("utf-8")
    ).hexdigest()
    out_path = FACE_CACHE / f"{key}.jpg"
    if not out_path.exists():
        Image.fromarray(face_rgb).save(out_path, quality=85)
    return key


def _run_scan_job(job_id: str, folder: Path) -> None:
    """背景執行掃描;進度寫入 JOBS[job_id]。"""
    job = JOBS[job_id]
    try:
        images = _list_scannable_images(folder)
        job["total"] = len(images)
        if not images:
            job["status"] = "done"
            job["results"] = {"Good": [], "Bad": [], "NoFace": []}
            return

        good: list[dict] = []
        bad: list[dict] = []
        noface: list[dict] = []
        errors: list[dict] = []
        t0 = time.time()

        for i, img_path in enumerate(images, 1):
            if job.get("cancel"):
                job["status"] = "cancelled"
                return
            job["current"] = i
            job["current_name"] = img_path.name
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.01)
            job["eta_seconds"] = (len(images) - i) / rate if rate > 0 else 0

            try:
                res = predict_image_quality(str(img_path))
                item = {
                    "path": str(img_path),
                    "name": img_path.name,
                    "mtime": img_path.stat().st_mtime,
                    "prob": float(res["probability"]),
                    "label": res["label"],
                }
                if res["label"] == "Bad":
                    item["face_id"] = _save_face_thumb(res["face_image"], img_path)
                    bad.append(item)
                else:
                    good.append(item)
            except NoFaceDetectedError:
                noface.append({
                    "path": str(img_path), "name": img_path.name,
                    "mtime": img_path.stat().st_mtime,
                })
            except Exception as e:
                errors.append({"path": str(img_path), "error": str(e)})

        # 按信心高到低排
        bad.sort(key=lambda x: x["prob"], reverse=True)
        good.sort(key=lambda x: x["prob"], reverse=True)

        job["status"] = "done"
        job["results"] = {
            "Good": good, "Bad": bad, "NoFace": noface,
            "errors": errors,
        }
        job["folder"] = str(folder)
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.post("/api/scan")
def start_scan(req: ScanRequest):
    folder = Path(req.folder).expanduser()
    if not folder.is_dir():
        raise HTTPException(404, f"資料夾不存在:{folder}")
    if not MODEL_PATH.is_file():
        raise HTTPException(503, "模型檔不存在,請先訓練或下載 models/mobilenet_face.pth")

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "status": "running",
        "current": 0,
        "total": 0,
        "current_name": "",
        "eta_seconds": 0,
        "results": None,
        "folder": str(folder),
    }
    Thread(target=_run_scan_job, args=(job_id, folder), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/scan/{job_id}")
def get_scan(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "找不到此 job")
    return job


@app.delete("/api/scan/{job_id}")
def cancel_scan(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "找不到此 job")
    job["cancel"] = True
    return {"ok": True}


# ──────────────────────────────────────────────────────────────────────────
# 縮圖服務 — 以 path 為 key 即時 resize + 快取
# ──────────────────────────────────────────────────────────────────────────
@app.get("/api/image")
def get_image(path: str = Query(...), w: int = Query(0)):
    """讀本機圖片,可選擇 resize 寬度。"""
    p = Path(path)
    if not p.is_file() or p.suffix.lower() not in VALID_EXTS:
        raise HTTPException(404, "圖片不存在或不支援格式")

    if w <= 0 or w >= 2000:
        # 原圖直送
        return FileResponse(str(p))

    # 縮圖(快取在 backend/cache/thumbs/{hash}_{w}.jpg)
    key = hashlib.md5(
        f"{p}::{p.stat().st_mtime}::{w}".encode("utf-8")
    ).hexdigest()
    cache_path = CACHE_DIR / "thumbs" / f"{key}.jpg"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        try:
            img = Image.open(p).convert("RGB")
            ratio = w / img.width
            new_size = (w, int(img.height * ratio))
            img.thumbnail(new_size, Image.LANCZOS)
            img.save(cache_path, quality=82, optimize=True)
        except Exception as e:
            raise HTTPException(500, f"縮圖失敗:{e}")
    return FileResponse(str(cache_path), media_type="image/jpeg")


@app.get("/api/face/{face_id}")
def get_face(face_id: str):
    """AI 裁切人臉縮圖(掃描時已存在 cache)。"""
    p = FACE_CACHE / f"{face_id}.jpg"
    if not p.is_file():
        raise HTTPException(404, "face 縮圖不存在")
    return FileResponse(str(p), media_type="image/jpeg")


# ──────────────────────────────────────────────────────────────────────────
# Trash
# ──────────────────────────────────────────────────────────────────────────
def _manifest_path(folder: Path) -> Path:
    return folder / "Trash" / "manifest.json"


@app.post("/api/trash")
def move_to_trash(req: TrashRequest):
    folder = Path(req.folder)
    trash_dir = folder / "Trash"
    trash_dir.mkdir(exist_ok=True)

    manifest: list[dict] = []
    moved: list[str] = []
    failed: list[str] = []

    for path_str in req.paths:
        src = Path(path_str)
        if not src.is_file():
            failed.append(path_str)
            continue
        try:
            dest = trash_dir / src.name
            if dest.exists():
                dest = trash_dir / (
                    f"{src.stem}_{datetime.now().strftime('%H%M%S')}{src.suffix}"
                )
            shutil.move(str(src), str(dest))
            manifest.append({
                "original_path": str(src),
                "trash_path": str(dest),
                "deleted_at": datetime.now().isoformat(),
            })
            moved.append(path_str)
        except Exception as e:
            failed.append(f"{path_str}: {e}")

    # 更新 manifest
    mf = _manifest_path(folder)
    existing = []
    if mf.exists():
        try:
            existing = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.extend(manifest)
    mf.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                  encoding="utf-8")

    return {"moved": moved, "failed": failed, "total_in_trash": len(existing)}


@app.get("/api/trash")
def list_trash(folder: str = Query(...)):
    """列出指定資料夾下 Trash 的所有照片(含縮圖 URL)。"""
    folder_p = Path(folder)
    mf = _manifest_path(folder_p)
    if not mf.is_file():
        return {"items": [], "total": 0}
    try:
        manifest = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return {"items": [], "total": 0}

    items = []
    for entry in manifest:
        trash_path = entry["trash_path"]
        orig_path = entry["original_path"]
        exists = Path(trash_path).is_file()
        items.append({
            "trash_path": trash_path,
            "original_path": orig_path,
            "name": Path(orig_path).name,
            "deleted_at": entry.get("deleted_at", ""),
            "exists": exists,
        })
    return {"items": items, "total": len(items)}


@app.post("/api/trash/restore")
def restore_from_trash(req: RestoreRequest):
    folder = Path(req.folder)
    mf = _manifest_path(folder)
    if not mf.is_file():
        return {"restored": 0, "failed": []}

    try:
        manifest = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return {"restored": 0, "failed": []}

    target_set = set(req.trash_paths or [])  # 空 = 全部還原
    restored = 0
    failed: list[str] = []
    remaining: list[dict] = []

    for entry in manifest:
        # 不在選取清單 → 保留在 Trash
        if target_set and entry["trash_path"] not in target_set:
            remaining.append(entry)
            continue

        src = Path(entry["trash_path"])
        dst = Path(entry["original_path"])
        if not src.is_file():
            failed.append(str(src))
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst = dst.parent / (
                    f"{dst.stem}_restored_"
                    f"{datetime.now().strftime('%H%M%S')}{dst.suffix}"
                )
            shutil.move(str(src), str(dst))
            restored += 1
        except Exception as e:
            failed.append(f"{src}: {e}")
            remaining.append(entry)

    mf.write_text(json.dumps(remaining, indent=2, ensure_ascii=False),
                  encoding="utf-8")
    return {"restored": restored, "failed": failed, "remaining": len(remaining)}


# ──────────────────────────────────────────────────────────────────────────
# Production: serve frontend build
# ──────────────────────────────────────────────────────────────────────────
if FRONTEND_DIST.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=str(FRONTEND_DIST), html=True),
        name="frontend",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
