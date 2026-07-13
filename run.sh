#!/usr/bin/env bash
# 啟動 AI 表情相簿管家
# 首次執行會自動建立 venv、安裝依賴、建置前端
set -e
cd "$(dirname "$0")"

# ── Python 虛擬環境 ──────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "[setup] 建立虛擬環境 .venv ..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip -q
  .venv/bin/pip install -r requirements.txt -q
fi

# ── 官方 MediaPipe 模型（下載後先驗證 SHA-256）────────────────────
.venv/bin/python scripts/download_models.py

# ── 前端建置 ─────────────────────────────────────────────────────────
FRONTEND_DIST="frontend/dist/index.html"
FRONTEND_FINGERPRINT_FILE="frontend/dist/.source-fingerprint"

frontend_source_fingerprint() {
  {
    find frontend/src -type f -print
    for file in \
      frontend/index.html \
      frontend/package.json \
      frontend/package-lock.json \
      frontend/vite.config.js
    do
      [ ! -f "$file" ] || printf '%s\n' "$file"
    done
  } | LC_ALL=C sort | while IFS= read -r file; do
    shasum -a 256 "$file"
  done | shasum -a 256 | awk '{print $1}'
}

FRONTEND_FINGERPRINT="$(frontend_source_fingerprint)"
STORED_FRONTEND_FINGERPRINT=""
if [ -f "$FRONTEND_FINGERPRINT_FILE" ]; then
  STORED_FRONTEND_FINGERPRINT="$(cat "$FRONTEND_FINGERPRINT_FILE")"
fi

if [ "${FORCE_FRONTEND_BUILD:-0}" = "1" ] \
  || [ ! -f "$FRONTEND_DIST" ] \
  || [ "$STORED_FRONTEND_FINGERPRINT" != "$FRONTEND_FINGERPRINT" ]; then
  echo "[setup] 建置前端..."
  if ! command -v node &>/dev/null; then
    echo "❌ 找不到 node，請先安裝 Node.js (https://nodejs.org/)"
    exit 1
  fi
  cd frontend && npm install -q && npm run build && cd ..
  FRONTEND_FINGERPRINT="$(frontend_source_fingerprint)"
  printf '%s\n' "$FRONTEND_FINGERPRINT" > "$FRONTEND_FINGERPRINT_FILE"
fi

# ── 啟動後端(同時服務前端靜態檔) ────────────────────────────────────
echo ""
echo "✅ 啟動成功 → http://localhost:8000"
echo "   按 Ctrl+C 停止"
echo ""
exec .venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
