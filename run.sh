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

# ── 前端建置 ─────────────────────────────────────────────────────────
FRONTEND_DIST="frontend/dist/index.html"
if [ ! -f "$FRONTEND_DIST" ]; then
  echo "[setup] 建置前端..."
  if ! command -v node &>/dev/null; then
    echo "❌ 找不到 node，請先安裝 Node.js (https://nodejs.org/)"
    exit 1
  fi
  cd frontend && npm install -q && npm run build && cd ..
fi

# ── 啟動後端(同時服務前端靜態檔) ────────────────────────────────────
echo ""
echo "✅ 啟動成功 → http://localhost:8000"
echo "   按 Ctrl+C 停止"
echo ""
exec .venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
