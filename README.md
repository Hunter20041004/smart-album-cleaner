# AI 表情相簿管家 — Smart Album Cleaner

Smart Album Cleaner 是在本機執行的照片整理工具。它使用 MobileNetV3 模型分析人臉表情品質，讓使用者檢視掃描結果，並以可還原的 Trash 流程整理照片。

目前唯一的應用程式架構是 **FastAPI + Vue 3**：`backend/main.py` 提供 REST API 並服務建置後的前端，`frontend/` 是 Vue 3 SPA。專案不再提供另一套 Web UI 入口。

## 功能範圍

- 選擇本機照片資料夾或檔案後啟動掃描工作。
- 在 Vue 介面輪詢掃描進度並檢視分類結果。
- 透過 Trash 清單執行軟刪除、還原或移至系統垃圾桶。
- 由 FastAPI 限制可存取的 Host、Origin 與已授權照片根目錄。
- 使用 `weights_only=True` 載入 PyTorch checkpoint。

## 安裝與啟動

需求：Python 3.11+、Node.js 20.19+ 或 22.12+，以及 Git。

### macOS / Linux

```bash
git clone https://github.com/Hunter20041004/smart-album-cleaner.git
cd smart-album-cleaner

mkdir -p models
curl -L https://github.com/Hunter20041004/smart-album-cleaner/releases/latest/download/mobilenet_face.pth \
  -o models/mobilenet_face.pth

./run.sh
```

`run.sh` 會建立 `.venv`、安裝 Python 依賴、下載並驗證 MediaPipe 模型、在需要時建置 Vue 前端，最後以 Uvicorn 啟動 `backend.main:app`。瀏覽器開啟 <http://localhost:8000>。

### Windows

```bat
git clone https://github.com/Hunter20041004/smart-album-cleaner.git
cd smart-album-cleaner

mkdir models
curl -L https://github.com/Hunter20041004/smart-album-cleaner/releases/latest/download/mobilenet_face.pth -o models/mobilenet_face.pth

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\download_models.py

cd frontend
npm install
npm run build
cd ..

.venv\Scripts\uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## 開發模式

後端：

```bash
.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

前端：

```bash
cd frontend
npm install
npm run dev
```

Vite 開發伺服器由 `frontend/vite.config.js` 將 API 請求轉送至本機 FastAPI。

## 技術架構

```text
Vue 3 SPA (frontend/)
        │ REST API
        ▼
FastAPI (backend/main.py)
        │
        ├── src/face_detector.py      MediaPipe 人臉偵測
        ├── src/predict_face.py       MobileNetV3 推論
        └── 本機照片與 Trash          授權根目錄內操作
```

Production build 由 FastAPI 同源服務；開發時則由 Vite 提供前端 hot reload。

## 專案結構

```text
backend/main.py             FastAPI 應用程式與 REST API
frontend/                   Vue 3 + Vite 前端
src/                        資料準備、訓練、推論與評估
scripts/download_models.py  MediaPipe 模型下載與雜湊驗證
tests/                      Python 測試與公開作品集契約
run.sh                      macOS / Linux 本機啟動腳本
```

## 自訓模型

資料準備、訓練與評估仍由 `src/` 下的命令列工具負責：

```bash
python -m src.prepare_dataset
python -m src.train_mobilenet --arch mobilenet_v3_large
python -m src.train_mobilenet --arch mobilenet_v3_large --finetune
python -m src.evaluate --model models/mobilenet_face.pth
```

訓練照片、處理後資料、Trash、快取與模型權重都應保留在本機，且不提交到 Git。

## 驗證

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
cd frontend && npm install && npm run build
```

## License

MIT — 詳見 [`LICENSE`](LICENSE)。
