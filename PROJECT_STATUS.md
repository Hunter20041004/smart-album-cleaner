# 🎭 AI 表情相簿管家 — 專案進度文件

> 最後更新:2026-04-30
> 目前最佳 val_acc:**76.3%**(MobileNetV3-Large)

---

## 📌 一句話介紹

用**自訓的深度學習模型**,從相簿挑出表情崩壞的照片(閉眼、嘴歪、模糊…),一鍵軟刪除。

---

## 🏗 整體架構

```
原始影片 ─┐
          ├─→ extract_frames.py ─→ datasets/raw/{Good,Bad}/
原始照片 ─┘                                 │
                                            ▼
                              prepare_dataset.py
                          (MediaPipe 雙模型偵測 + 裁切 224x224)
                                            │
                                            ▼
                              datasets/processed/{Good,Bad}/
                                            │
                                            ▼
                              train_mobilenet.py
                  (Stage-1 凍結特徵 → Stage-2 解凍微調 → 閾值調校)
                                            │
                                            ▼
                              models/mobilenet_face.pth
                                            │
                                            ▼
                              predict_face.py ←─ app.py (Streamlit UI)
                              (TTA + 閾值 + Lazy 快取)
```

---

## ✅ 已完成功能

### 1. 資料準備工具

| 檔案 | 功能 |
|---|---|
| `src/extract_frames.py` | 從 mp4/mov 影片每 N 幀擷取一張 jpg |
| `src/prepare_dataset.py` | MediaPipe 雙模型偵測人臉 → 向外擴張 15% → 裁成 224×224 → 分到 Good/Bad |

**雙模型 fallback 機制**(避免漏臉):
- Primary: `model_selection=0` (短距離,擅長肖像)
- Fallback: `model_selection=1` (長距離,擅長合照中的人物)
- 修正前漏失 27 張,修正後降到 **3 張**

### 2. 訓練模組(`src/train_mobilenet.py`)

| 功能 | 設定 |
|---|---|
| 支援架構 | MobileNetV2 / **MobileNetV3-Large**(預設) |
| Stage-1 | 凍結 features,訓練 classifier(15 epochs, lr=1e-3) |
| Stage-2 | `--finetune` 解凍最後 3 個 blocks(5 epochs, lr=1e-5) |
| 資料增強 | `RandomResizedCrop` + `ColorJitter` + `RandomErasing` |
| 正則化 | Weight decay + Label smoothing(0.05) |
| 自動閾值調校 | 訓練結束掃描 0.30~0.70 找最佳 P(Bad) 切點 |
| 防呆 | finetune 時偵測 checkpoint 架構是否一致 |
| Device | 自動選 CUDA / **MPS(M2)** / CPU |

**Checkpoint 內容**:
```python
{
    "state_dict": ..., "arch": "mobilenet_v3_large",
    "classes": ["Bad", "Good"], "img_size": 224,
    "epoch": 9, "val_acc": 0.763,
    "decision_threshold": 0.47, "threshold_val_acc": 0.763,
}
```

### 3. 推論模組(`src/predict_face.py`)

```python
result = predict_image_quality("photo.jpg")
# {
#   "label": "Good" | "Bad",
#   "probability": 0.7234,
#   "probs": {"Bad": 0.28, "Good": 0.72},
#   "face_image": np.ndarray(224,224,3),
#   "decision_threshold": 0.47,
# }
```

| 功能 | 細節 |
|---|---|
| 雙模型 fallback face detection | 與 prepare_dataset 對齊,避免 train-serve skew |
| TTA(Test-Time Augmentation) | 水平翻轉 + 原圖,平均機率 |
| 閾值套用 | 用 checkpoint 寫入的 best threshold,不用死板 0.5 |
| Lazy singleton 快取 | 模型 / detector / transform 只載入一次 |
| 自訂例外 | `NoFaceDetectedError` 給 UI 用 try/except 跳過 |
| 多輸入支援 | `str` / `Path` / `np.ndarray (BGR)` |

### 4. Web UI(`app.py`)

- **Sidebar**:藍色「啟動 AI 深度掃描」按鈕、模型未訓練的防呆提示
- **歡迎頁**:使用步驟 + 分類規則表格
- **三分頁**: ⚠️ Bad / ✅ Good / 👀 NoFace
- **Bad 卡片**:原圖縮圖 + AI 鎖定區域(裁切後人臉) + 崩壞機率
- **批次軟刪除**:全選/全清 + 雙確認 + 移到 `Trash/` + manifest.json 紀錄
- **錯誤紀錄**:Sidebar Expander 顯示掃描失敗檔案
- **縮圖快取**:`@st.cache_data` 以 (path, mtime) 為鍵

---

## 📊 實驗紀錄

| 實驗 | val_acc | 備註 |
|---|---|---|
| V2 + 單模型偵測 + 基礎增強 | 76.1% | 漏失 27 張困難樣本 |
| V2 + 單模型 + finetune | 77.3% | 過擬合明顯 |
| V2 + 強化增強 + finetune | 79.0% | 數字虛胖,因為困難樣本還沒進來 |
| V2 + **雙模型** + 完整 pipeline | 72.3% | **真實能力**,因為包含困難樣本 |
| **V3-Large + 雙模型 + 完整 pipeline** | **76.3%** ⭐ | 目前最佳 |

**結論**:
- SE Attention 對人臉細節有用(+4%)
- 雙模型偵測讓 27 張肖像照不再被漏掉
- 主要瓶頸已從「模型架構」轉為「資料規模」

---

## 📂 專案結構

```
smart-album-cleaner/
├── README.md
├── PROJECT_STATUS.md             ← 本檔
├── requirements.txt
├── run.sh
├── app.py                        ← Streamlit UI
├── datasets/
│   ├── raw/                      ← 你丟原始照片進這裡
│   │   ├── Good/
│   │   └── Bad/
│   └── processed/                ← MediaPipe 裁切結果(自動產生)
│       ├── Good/  (409 張)
│       └── Bad/   (477 張)
├── models/
│   ├── mobilenet_face.pth        ← 目前模型(V3-Large, val_acc 76.3%)
│   └── mobilenet_v2_backup.pth   ← V2 備份(val_acc 72.3%)
└── src/
    ├── extract_frames.py
    ├── prepare_dataset.py
    ├── train_mobilenet.py
    └── predict_face.py
```

---

## ⚙️ 技術棧

| 類別 | 工具 |
|---|---|
| 深度學習框架 | PyTorch 2.x + torchvision |
| 模型架構 | MobileNetV3-Large(ImageNet 預訓 + 自訓 head) |
| 人臉偵測 | MediaPipe 0.10.9(雙模型 fallback) |
| 影像處理 | OpenCV(cv2) + Pillow |
| Web UI | Streamlit |
| 加速 | Apple MPS(Metal Performance Shaders) |

---

## 🚀 標準工作流程

```bash
# 一次性:訓練模型(已完成)
python -m src.prepare_dataset
python -m src.train_mobilenet --arch mobilenet_v3_large

# 日常使用:開 UI
./run.sh   # 或 .venv/bin/streamlit run app.py

# 進階:加資料後重訓
python -m src.prepare_dataset            # 重新裁切
python -m src.train_mobilenet            # 重訓 stage-1
python -m src.train_mobilenet --finetune # 可選:再 finetune
```

---

## 📝 待辦清單

### 🔥 高優先(會直接影響準確率與分數)

- [ ] **加資料**:多收集 200-500 張照片,特別是「邊界模糊的灰色地帶」
  - 預期:val_acc 76% → **82-85%**
  - 工時:你 30-60 分鐘蒐集
- [ ] **資料清洗**:人工檢視 `datasets/raw/` 有沒有標錯
  - 預期:+2~3%
  - 工時:30 分鐘
- [ ] **移除 3 張 NoFace 樣本**:`IMG_2691.PNG` / `IMG_2886.PNG` / `IMG_2909.PNG`
  - 雙模型也抓不到,留著只是雜訊

### 📚 中優先(報告 / 口試需要)

- [ ] **撰寫專案報告**
  - 動機與目標
  - 架構圖 + 流程圖
  - 實驗對比(V2 vs V3 表格 / 雙模型 vs 單模型)
  - 錯誤案例分析(NoFace 的 3 張、Bad 誤判邊界案例)
- [ ] **製作 confusion matrix** — `sklearn.metrics.confusion_matrix` + matplotlib heatmap
- [ ] **產出訓練 loss/acc 曲線圖** — 給報告當圖
- [ ] **準備口試 demo 流程** — 預備幾張會 work、幾張會失敗的範例

### 💡 低優先(體驗 / 加分項)

- [ ] **UI 加入「不確定」帶**:`probability` 在 0.45-0.55 時用黃色標示「🤔 AI 不確定」
- [ ] **加「還原 Trash」按鈕**:讀 `manifest.json` 把照片搬回原位
- [ ] **支援多資料夾批次處理**:一次掃多個路徑
- [ ] **更新 README.md**:把現狀寫進去,給組員 / 老師看

### 🔬 研究 / 實驗(時間有的話)

- [ ] **From-scratch baseline 對比**:`weights=None` 訓一個版本,證明預訓的價值
- [ ] **EfficientNet-B0 對比**:更強 backbone 看會不會繼續上升
- [ ] **Mixup / CutMix 增強**:對小資料集通常有效
- [ ] **Cosine LR Schedule**:替代固定 lr,可能再榨 1~2%

---

## ⚠️ 已知限制

| 問題 | 影響 | 暫無解 |
|---|---|---|
| 訓練資料量小(709 張 train) | val_acc 天花板約 80~85% | 加資料是唯一解 |
| 極暗 / 電視截圖 / 遠距人物 | 雙模型仍可能漏(剩 3 張) | 等 MediaPipe 更新或換 RetinaFace |
| 「邊界表情」誤判率高 | ~24% 誤判,不少在 0.45-0.55 機率帶 | 加 UI「不確定」標示 + 加邊界資料 |
| 模型不認識「我們沒拍過的人」 | 對家人/朋友以外的人臉判得差 | 屬於 generalization 問題,需更廣資料 |

---

## 🗺 可能的演進路線

```
目前:76.3%
   │
   ├─[加 300 張資料]──→ 82-85%  ⭐ 推薦下一步
   │       │
   │       ├─[+ 資料清洗]──→ 84-87%
   │       │
   │       └─[+ EfficientNet-B0]──→ 85-88%
   │
   └─[只調模型不加資料]──→ 76-78%(撞牆)
```

---

## 🎓 給報告/口試的講法草稿

> 我們開發了一個基於**深度學習遷移學習**的相簿表情分類系統。
>
> **資料前處理**使用 MediaPipe 的雙模型 fallback 策略(short-range + full-range)來定位人臉,並向外擴張 15% 保留完整頭部,統一裁切為 224×224。
>
> **模型架構**採用 **MobileNetV3-Large**(含 Squeeze-and-Excitation Attention),凍結 ImageNet 預訓特徵層,訓練自定義二元分類 head。第二階段 fine-tune 解凍最後 3 個 inverted residual blocks,使用 lr=1e-5 配合 weight decay 1e-4 防止過擬合。
>
> **架構選型**過程中我們對比了 MobileNetV2 與 V3-Large,V3-Large 因 SE Attention 在細粒度人臉特徵任務上**領先 4 個百分點**(72.3% → 76.3%),驗證了注意力機制對此任務的有效性。
>
> **推論時**採用 Test-Time Augmentation(水平翻轉求平均)以及訓練時掃描出的最佳決策閾值,而非死板的 0.5。
>
> 系統最終 validation accuracy 為 **76.3%**,具備 Streamlit Web UI 提供批次掃描、視覺化分類結果、軟刪除等完整功能。
