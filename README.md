# AI 表情相簿管家 — Smart Album Cleaner

> 用**自訓的 MobileNetV3 深度學習模型**,在 60 秒內從你的相簿挑出表情崩壞的廢片(閉眼、嘴歪、模糊…),一鍵軟刪除。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Vue 3](https://img.shields.io/badge/Frontend-Vue%203-42b883.svg)](https://vuejs.org/)

---

## ✨ 為什麼用這個?

- ⚡ **本機運行** — 首次安裝模型後可離線使用，你的照片不會上傳到雲端
- 🧠 **自訓深度學習模型** — MobileNetV3-Large + SE Attention,專為「人臉表情品質」訓練
- 🎯 **三分頁清晰分類** — ⚠️ 建議刪除 / ✅ 完美表情 / 👀 未偵測到人臉
- 🛡️ **軟刪除安全網** — 移到 `Trash/` 而非永久刪除,可隨時還原
- 🍎 **Apple Silicon 加速** — 自動使用 MPS,M2 上掃 1000 張 < 2 分鐘
- 🔌 **前後端分離** — FastAPI REST API + Vue 3 SPA,介面流暢無卡頓

---

## 🚀 安裝與啟動

### 事前準備

| 工具 | 版本 | 下載 |
|------|------|------|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 20.19+ 或 22.12+ | [nodejs.org](https://nodejs.org/) |
| Git | 任意 | [git-scm.com](https://git-scm.com/) |

---

### Mac / Linux

```bash
# 1. 下載專案
git clone https://github.com/Hunter20041004/smart-album-cleaner.git
cd smart-album-cleaner

# 2. 下載預訓模型權重(免自己訓練!)
mkdir -p models
curl -L https://github.com/Hunter20041004/smart-album-cleaner/releases/latest/download/mobilenet_face.pth \
    -o models/mobilenet_face.pth

# 3. 建置前端
cd frontend && npm install && npm run build && cd ..

# 4. 啟動(首次會自動建虛擬環境 + 安裝套件,約 1-2 分鐘)
./run.sh
```

瀏覽器開啟 → **http://localhost:8000**

---

### Windows

```bat
:: 1. 下載專案
git clone https://github.com/Hunter20041004/smart-album-cleaner.git
cd smart-album-cleaner

:: 2. 下載預訓模型權重
mkdir models
curl -L https://github.com/Hunter20041004/smart-album-cleaner/releases/latest/download/mobilenet_face.pth -o models/mobilenet_face.pth

:: 3. 建置前端
cd frontend
npm install
npm run build
cd ..

:: 4. 建立虛擬環境 + 安裝套件
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\download_models.py

:: 5. 啟動
.venv\Scripts\uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

瀏覽器開啟 → **http://localhost:8000**

---

## 📖 使用教學

### Step 1 — 選擇相片資料夾

啟動後在首頁點「選擇資料夾」,輸入你的相片資料夾路徑(例如 `/Users/yourname/Photos/2024`),支援多層子資料夾。

### Step 2 — 開始 AI 掃描

點「開始掃描」,畫面會顯示即時進度條與預估剩餘時間。掃描速度約每張 80-130ms(M2 MacBook Air)。

### Step 3 — 查看分析結果

掃描完成後進入結果頁,分為三個分頁：

| 分頁 | 說明 |
|------|------|
| ⚠️ **建議刪除** | AI 判定表情崩壞(閉眼、嘴歪等),附「原圖」與「AI 鎖定區」對照 |
| ✅ **完美表情** | AI 判定表情自然良好 |
| 👀 **未偵測到人臉** | 無人臉或人臉過小無法判定(風景、物品等) |

### Step 4 — 勾選並刪除廢片

在「建議刪除」頁：
- 點「全選」快速選取所有廢片,或手動勾選想刪的張數
- 點「移到 Trash」— 照片**不會永久消失**,而是移到專案的 `Trash/` 資料夾

### Step 5 — 還原誤刪的照片(選用)

如果不小心刪錯了：
1. 點左側選單「♻ Trash」
2. 勾選想還原的照片
3. 點「還原勾選的 N 張」— 照片會回到原來的資料夾

---

## 🧠 技術架構

```
原始照片
     │
     ▼
┌─────────────────────────────┐
│  prepare_dataset.py         │
│  ─ MediaPipe Tasks API      │  ← 官方模型 + letterbox fallback
│  ─ 向外擴張 15% 保留頭部    │
│  ─ 統一裁成 224×224         │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  train_mobilenet.py         │
│  ─ MobileNetV3-Large + SE   │  ← ImageNet 預訓 backbone
│  ─ Stage-1 凍結 + Stage-2   │
│  ─ Augmentation + WeightDecay│
│  ─ 自動掃決策閾值            │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐    ┌────────────────────────┐
│  FastAPI 後端               │◄──►│  Vue 3 前端(SPA)       │
│  ─ REST API (8 endpoints)   │    │  ─ 掃描進度 polling     │
│  ─ 非同步掃描 job queue     │    │  ─ 三分頁結果 + 勾選   │
│  ─ 縮圖快取(SHA-256)       │    │  ─ Trash 管理 + 還原   │
│  ─ 同源服務靜態檔           │    │  ─ Darkroom 視覺主題   │
└─────────────────────────────┘    └────────────────────────┘
```

**效能**(M2 MacBook Air):

| 任務 | 耗時 |
|------|------|
| MediaPipe 偵測 + 裁切(1 張) | ~50 ms |
| MobileNetV3 推論(含 TTA) | ~80 ms |
| **完整掃描 1000 張** | **~2 分鐘** |
| Stage-1 訓練(15 epochs, 1000 樣本) | ~70 秒 |

---

## 📂 專案結構

```
smart-album-cleaner/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── run.sh                        ← 一鍵啟動(Mac/Linux)
├── backend/
│   └── main.py                   ← FastAPI app (REST API + 靜態服務)
├── frontend/
│   ├── src/
│   │   ├── App.vue               ← 根元件 + 全域狀態
│   │   ├── api.js                ← API 呼叫封裝
│   │   ├── styles.css            ← Darkroom 視覺主題
│   │   └── components/
│   │       ├── ScanProgress.vue  ← 掃描進度條
│   │       ├── ResultsView.vue   ← 三分頁結果
│   │       └── TrashView.vue     ← Trash 管理
│   ├── package.json
│   └── vite.config.js
├── src/
│   ├── extract_frames.py         ← 影片轉影格
│   ├── face_detector.py          ← MediaPipe Tasks 安全轉接層
│   ├── prepare_dataset.py        ← MediaPipe 裁切前處理
│   ├── train_mobilenet.py        ← 訓練腳本
│   ├── predict_face.py           ← 推論(TTA + 快取)
│   └── evaluate.py               ← 獨立評估腳本
├── tests/
├── datasets/
│   ├── raw/{Good,Bad}/           ← 你的原始照片(不 commit)
│   └── processed/{Good,Bad}/    ← 自動產生(不 commit)
└── models/
    ├── mobilenet_face.pth        ← 從 Releases 下載(不 commit)
    └── blaze_face_short_range.tflite ← 官方模型、SHA-256 驗證(不 commit)
```

---

## ⚙️ 進階:自訓你自己的模型

如果想用自己的照片重新訓練個性化模型：

```bash
# 1. 把照片分好類丟進這兩個資料夾
datasets/raw/Good/    # 表情自然的照片(建議 150 張以上)
datasets/raw/Bad/     # 表情崩壞的照片(建議 150 張以上)

# 2. 裁切 → 訓練 → 微調
python -m src.prepare_dataset
python -m src.train_mobilenet --arch mobilenet_v3_large
python -m src.train_mobilenet --arch mobilenet_v3_large --finetune

# 評估模型成效
python -m src.evaluate --model models/mobilenet_face.pth
```

---

## 🎯 路線圖

### 已完成 ✅
- [x] MediaPipe Tasks 人臉偵測 + letterbox fallback
- [x] MobileNetV2 / V3-Large 雙架構支援
- [x] Stage-1(凍結) + Stage-2(解凍微調)兩階段訓練
- [x] 資料增強 + 決策閾值自動調校 + Test-Time Augmentation
- [x] FastAPI 後端 + Vue 3 前端分離架構
- [x] 非同步掃描 + 即時進度 polling
- [x] 三分頁 UI + 批次軟刪除 + Trash 還原
- [x] 伺服器端縮圖快取(Pillow + SHA-256)

### 未來 🔮
- [ ] 連拍智慧選最佳(同時間組內取 Good 信心最高者)
- [ ] EXIF 整合(按相機型號 / 拍攝日期篩選)
- [ ] 桌面 App 打包(`.app` / `.exe`)
- [ ] 使用者標註回饋,建立 data flywheel

---

## 🔐 本機資料與安全設計

- API 只綁定 `127.0.0.1`，並拒絕非 loopback Host 與未允許的瀏覽器 Origin。
- 使用者選取資料夾後，後端才會在本次程序授權該 canonical root；預覽、Trash、還原與系統垃圾桶都不能越界，symlink 也不能跳脫。
- 照片與人臉縮圖只在本機處理，`datasets/`、`Trash/`、`backend/cache/` 與模型權重均由 `.gitignore` 排除。
- PyTorch checkpoint 一律使用 `weights_only=True` 載入，不允許任意 pickle 程式碼執行。
- MediaPipe Tasks 模型取自官方儲存空間；下載器驗證 SHA-256：`b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f`。

重現本專案的安全檢查：

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check backend src tests scripts
.venv/bin/pip-audit
.venv/bin/bandit -q -r backend src scripts
cd frontend && npm ci && npm run build && npm audit --audit-level=high
```

---

## 🤝 貢獻

歡迎 PR!特別需要:
- 更多元的訓練資料(不同年齡 / 種族 / 光線)
- 桌面 App 打包(Windows / Linux 測試)
- 翻譯(英文 / 日文 / 韓文)

---

## 📜 License

MIT — 詳見 [`LICENSE`](LICENSE)

---

## 🙏 致謝

- [PyTorch](https://pytorch.org/) + [torchvision](https://pytorch.org/vision/) — MobileNetV3 預訓權重
- [MediaPipe](https://github.com/google/mediapipe) — 人臉偵測
- [FastAPI](https://fastapi.tiangolo.com/) — 後端框架
- [Vue 3](https://vuejs.org/) + [Vite](https://vitejs.dev/) — 前端框架
