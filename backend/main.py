"""
FastAPI 後端 — AI 表情相簿管家 v2.0

啟動: uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
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
    folder: str | None = None
    paths: list[str] | None = None


class TrashRequest(BaseModel):
    folder: str  # 掃描的根資料夾
    paths: list[str]  # 要刪除的絕對路徑


class RestoreRequest(BaseModel):
    folder: str
    trash_paths: list[str] | None = None  # None = 全部還原


class SystemTrashRequest(BaseModel):
    folder: str
    trash_paths: list[str] | None = None  # None = 全部移到系統垃圾桶


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


@app.post("/api/select-folder")
def select_folder():
    """Open a native folder picker on the local machine and return its path."""
    system = platform.system()
    if system == "Darwin":
        script = (
            'POSIX path of (choose folder with prompt '
            '"選擇要掃描的照片資料夾")'
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                return JSONResponse({"cancelled": True, "folder": None})
            raise HTTPException(500, e.stderr.strip() or "無法開啟資料夾選擇器")
        folder = result.stdout.strip().rstrip("/")
        return {"cancelled": False, "folder": folder}

    raise HTTPException(501, "目前僅支援 macOS 原生資料夾選擇器")


@app.post("/api/select-files")
def select_files():
    """Open a native multi-file picker and return selected image paths."""
    system = platform.system()
    if system == "Darwin":
        script = (
            'set chosenFiles to choose file with prompt "選擇要掃描的照片" '
            'of type {"public.image"} with multiple selections allowed\n'
            'set output to ""\n'
            'repeat with chosenFile in chosenFiles\n'
            '  set output to output & POSIX path of chosenFile & linefeed\n'
            'end repeat\n'
            'return output'
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                return JSONResponse({"cancelled": True, "paths": []})
            raise HTTPException(500, e.stderr.strip() or "無法開啟照片選擇器")
        paths = [p for p in result.stdout.splitlines() if p.strip()]
        return {"cancelled": False, "paths": paths}

    raise HTTPException(501, "目前僅支援 macOS 原生檔案選擇器")


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


def _scan_root_for_files(images: list[Path]) -> Path:
    """Use a stable root folder for selected files and Trash manifest storage."""
    parents = [str(p.parent) for p in images]
    return Path(os.path.commonpath(parents))


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


def _run_scan_job(job_id: str, folder: Path, images: list[Path] | None = None) -> None:
    """背景執行掃描;進度寫入 JOBS[job_id]。"""
    job = JOBS[job_id]
    try:
        images = images if images is not None else _list_scannable_images(folder)
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
    selected_images: list[Path] | None = None
    if req.paths:
        selected_images = []
        for path_str in req.paths:
            img = Path(path_str).expanduser()
            if not img.is_file():
                raise HTTPException(404, f"圖片不存在:{img}")
            if img.suffix.lower() not in VALID_EXTS:
                raise HTTPException(400, f"不支援的圖片格式:{img.name}")
            selected_images.append(img)
        folder = _scan_root_for_files(selected_images)
    elif req.folder:
        folder = Path(req.folder).expanduser()
        if not folder.is_dir():
            raise HTTPException(404, f"資料夾不存在:{folder}")
    else:
        raise HTTPException(400, "請先選擇資料夾或圖片")

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
    Thread(target=_run_scan_job, args=(job_id, folder, selected_images), daemon=True).start()
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


def _read_manifest(folder: Path) -> list[dict]:
    mf = _manifest_path(folder)
    if not mf.exists():
        return []
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_manifest(folder: Path, manifest: list[dict]) -> None:
    mf = _manifest_path(folder)
    mf.parent.mkdir(exist_ok=True)
    mf.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                  encoding="utf-8")


def _dedupe_manifest_entries(manifest: list[dict]) -> list[dict]:
    """Keep one visible Trash entry for the same original file or same image copy."""
    deduped: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for entry in sorted(
        manifest,
        key=lambda x: x.get("deleted_at", ""),
        reverse=True,
    ):
        trash_path = Path(entry.get("trash_path", ""))
        original_path = Path(entry.get("original_path", ""))
        if not trash_path.is_file():
            continue

        try:
            stat = trash_path.stat()
            file_key = (original_path.name or trash_path.name, str(stat.st_size))
        except OSError:
            continue

        original_key = ("original", str(original_path))
        content_key = ("file", "::".join(file_key))
        if original_key in seen_keys or content_key in seen_keys:
            continue

        seen_keys.add(original_key)
        seen_keys.add(content_key)
        deduped.append(entry)
    return list(reversed(deduped))


def _move_file_to_system_trash(path: Path) -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("目前僅支援 macOS 系統垃圾桶")
    script = (
        'on run argv\n'
        '  tell application "Finder"\n'
        '    delete (POSIX file (item 1 of argv) as alias)\n'
        '  end tell\n'
        'end run'
    )
    subprocess.run(
        ["osascript", "-e", script, "--", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


@app.post("/api/trash")
def move_to_trash(req: TrashRequest):
    folder = Path(req.folder)
    trash_dir = folder / "Trash"
    trash_dir.mkdir(exist_ok=True)

    existing = _dedupe_manifest_entries(_read_manifest(folder))
    active_keys: set[tuple[str, str]] = set()
    for entry in existing:
        trash_path = Path(entry["trash_path"])
        original_path = Path(entry["original_path"])
        try:
            active_keys.add((
                original_path.name or trash_path.name,
                str(trash_path.stat().st_size),
            ))
        except OSError:
            pass

    new_entries: list[dict] = []
    moved: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []

    for path_str in req.paths:
        src = Path(path_str)
        if not src.is_file():
            failed.append(path_str)
            continue
        try:
            if trash_dir in src.parents:
                skipped.append(f"{path_str}: 已在 Trash")
                continue
            src_key = (src.name, str(src.stat().st_size))
            if src_key in active_keys:
                skipped.append(f"{path_str}: 已存在 Trash")
                continue

            dest = trash_dir / src.name
            if dest.exists():
                dest = trash_dir / (
                    f"{src.stem}_{datetime.now().strftime('%H%M%S')}{src.suffix}"
                )
            shutil.move(str(src), str(dest))
            entry = {
                "original_path": str(src),
                "trash_path": str(dest),
                "deleted_at": datetime.now().isoformat(),
            }
            new_entries.append(entry)
            active_keys.add((dest.name, str(dest.stat().st_size)))
            moved.append(path_str)
        except Exception as e:
            failed.append(f"{path_str}: {e}")

    existing.extend(new_entries)
    existing = _dedupe_manifest_entries(existing)
    _write_manifest(folder, existing)

    return {
        "moved": moved,
        "failed": failed,
        "skipped": skipped,
        "total_in_trash": len(existing),
    }


@app.get("/api/trash")
def list_trash(folder: str = Query(...)):
    """列出指定資料夾下 Trash 的所有照片(含縮圖 URL)。"""
    folder_p = Path(folder)
    manifest = _dedupe_manifest_entries(_read_manifest(folder_p))
    _write_manifest(folder_p, manifest)

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


@app.post("/api/trash/system-delete")
def move_trash_items_to_system_trash(req: SystemTrashRequest):
    folder = Path(req.folder)
    mf = _manifest_path(folder)
    if not mf.is_file():
        return {"deleted": 0, "failed": [], "remaining": 0}

    try:
        manifest = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return {"deleted": 0, "failed": [], "remaining": 0}

    target_set = set(req.trash_paths or [])  # 空 = 全部移到系統垃圾桶
    deleted = 0
    failed: list[str] = []
    remaining: list[dict] = []

    for entry in manifest:
        trash_path = entry["trash_path"]
        if target_set and trash_path not in target_set:
            remaining.append(entry)
            continue

        src = Path(trash_path)
        if not src.is_file():
            deleted += 1
            continue
        try:
            _move_file_to_system_trash(src)
            deleted += 1
        except subprocess.CalledProcessError as e:
            failed.append(f"{src}: {e.stderr.strip() or e}")
            remaining.append(entry)
        except Exception as e:
            failed.append(f"{src}: {e}")
            remaining.append(entry)

    mf.write_text(json.dumps(remaining, indent=2, ensure_ascii=False),
                  encoding="utf-8")
    return {"deleted": deleted, "failed": failed, "remaining": len(remaining)}


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
