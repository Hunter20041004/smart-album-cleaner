# AI 表情相簿管家 — Smart Album Cleaner

> 用**自訓的 MobileNetV3 深度學習模型**,在 60 秒內從你的相簿挑出表情崩壞的廢片(閉眼、嘴歪、模糊…),一鍵軟刪除。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Vue 3](https://img.shields.io/badge/Frontend-Vue%203-42b883.svg)](https://vuejs.org/)

---

## ✨ 為什麼用這個?

- ⚡ **本機運行,完全離線** — 你的私密照片不會上傳到任何雲端
- 🧠 **自訓深度學習模型** — MobileNetV3-Large + SE Attention,專為「人臉表情品質」訓練
- 🎯 **三分頁清晰分類** — ⚠️ 建議刪除 / ✅ 完美表情 / 👀 未偵測到人臉
- 🛡️ **軟刪除安全網** — 移到 `Trash/` 而非永久刪除,可隨時還原
- 🍎 **Apple Silicon 加速** — 自動使用 MPS,M2 上掃 1000 張 < 2 分鐘
- 🔌 **前後端分離** — FastAPI REST API + Vue 3 SPA,介面流暢無卡頓

---

## 🚀 快速開始

### 方式 A:下載即用(推薦,5 分鐘)

```bash
# 1. clone 專案
git clone https://github.com/cengweiting/smart-album-cleaner.git
cd smart-album-cleaner

# 2. 建虛擬環境 + 安裝 Python 依賴
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. 下載預訓好的模型權重(免訓練!)
mkdir -p models
curl -L https://github.com/cengweiting/smart-album-cleaner/releases/latest/download/mobilenet_face.pth \
    -o models/mobilenet_face.pth

# 4. 建置前端
cd frontend && npm install && npm run build && cd ..

# 5. 啟動後端 → 瀏覽器開啟 http://localhost:8000
./run.sh
```

> 📦 **預訓權重檔**:`mobilenet_face.pth`(12 MB,val_acc 75.1%)
> 🔗 下載:[GitHub Releases](https://github.com/cengweiting/smart-album-cleaner/releases)

### 方式 B:開發模式(前後端分離)

```bash
# 終端機 1 — 後端
source .venv/bin/activate
uvicorn backend.main:app --reload --port 8000

# 終端機 2 — 前端(Hot reload)
cd frontend && npm run dev
# 開啟 http://localhost:5173
```

### 方式 C:自訓你自己的模型

```bash
# 1. 把照片分好類丟進這兩個資料夾
datasets/raw/Good/    # 表情自然的照片(150 張以上)
datasets/raw/Bad/     # 表情崩壞的照片(150 張以上)

# 2. 裁切 → 訓練 → 微調
python -m src.prepare_dataset
python -m src.train_mobilenet --arch mobilenet_v3_large
python -m src.train_mobilenet --arch mobilenet_v3_large --finetune

# 3. 啟動
./run.sh
```

---

## 🧠 技術架構

```
原始照片
     │
     ▼
┌─────────────────────────────┐
│  prepare_dataset.py         │
│  ─ MediaPipe 三層 Fallback  │  ← short-range → full-range → letterbox
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
│  ─ 縮圖快取(MD5)           │    │  ─ Trash 管理 + 還原   │
│  ─ 同源服務靜態檔           │    │  ─ Darkroom 視覺主題   │
└─────────────────────────────┘    └────────────────────────┘
```

**效能**(M2 MacBook Air):

| 任務 | 耗時 |
|---|---|
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
├── run.sh                        ← 一鍵啟動
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
│   ├── prepare_dataset.py        ← MediaPipe 裁切前處理
│   ├── train_mobilenet.py        ← 訓練腳本
│   ├── predict_face.py           ← 推論(TTA + 快取)
│   └── evaluate.py               ← 獨立評估腳本
├── tests/
├── datasets/
│   ├── raw/{Good,Bad}/           ← 你的原始照片(不 commit)
│   └── processed/{Good,Bad}/    ← 自動產生(不 commit)
└── models/
    └── mobilenet_face.pth        ← 從 Releases 下載(不 commit)
```

---

## ⚙️ 進階指令

```bash
# 用 V2 backbone 對比實驗
python -m src.train_mobilenet --arch mobilenet_v2

# 評估現有模型(不重訓)
python -m src.evaluate --model models/mobilenet_face.pth

# 跑測試
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## 🎯 路線圖

### 已完成 ✅
- [x] MediaPipe 三層 fallback 人臉偵測
- [x] MobileNetV2 / V3-Large 雙架構支援
- [x] Stage-1(凍結) + Stage-2(解凍微調)兩階段訓練
- [x] 資料增強 + 決策閾值自動調校 + Test-Time Augmentation
- [x] FastAPI 後端 + Vue 3 前端分離架構
- [x] 非同步掃描 + 即時進度 polling
- [x] 三分頁 UI + 批次軟刪除 + Trash 還原
- [x] 伺服器端縮圖快取(Pillow + MD5)

### 未來 🔮
- [ ] 連拍智慧選最佳(同時間組內取 Good 信心最高者)
- [ ] EXIF 整合(按相機型號 / 拍攝日期篩選)
- [ ] 桌面 App 打包(`.app` / `.exe`)
- [ ] 使用者標註回饋,建立 data flywheel

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
