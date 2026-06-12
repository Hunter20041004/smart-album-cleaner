"""
AI 表情相簿管家 — Streamlit Web UI · DARKROOM EDITION v0.5

設計參考:Adobe Lightroom CC · NYT Magazine · Linear app
啟動: ./run.sh  或  streamlit run app.py
"""
from __future__ import annotations

import html
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────
# 0. 模組載入
# ──────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.predict_face import predict_image_quality, NoFaceDetectedError
except ImportError:
    st.error("⚠ 找不到 src/predict_face.py")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────
# 1. 常數與頁面設定
# ──────────────────────────────────────────────────────────────────────────
APP_ROOT = Path(__file__).parent
MODEL_PATH = APP_ROOT / "models" / "mobilenet_face.pth"
RECENT_FOLDERS_FILE = APP_ROOT / ".streamlit" / "recent_folders.json"
RECENT_FOLDERS_MAX = 8
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

st.set_page_config(
    page_title="Darkroom · AI 表情管家",
    page_icon="🎞",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "**Darkroom · AI 表情管家** v0.5.0\n\n"
                 "MIT License · MobileNetV3-Large + MediaPipe",
    },
)


# ──────────────────────────────────────────────────────────────────────────
# 2. 設計系統(精煉版)
# ──────────────────────────────────────────────────────────────────────────
def inject_theme() -> None:
    st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;600;700;800;900&family=Noto+Serif+TC:wght@600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

    <style>
    :root {
        /* 背景:暖米 + 白(與海報主視覺一致)*/
        --bg-0:         #f0ead8;        /* 最深(sidebar / header) */
        --bg-1:         #faf6ea;        /* 主背景 */
        --bg-2:         #ffffff;        /* 卡片底 */
        --bg-3:         #f5f0e0;        /* 浮起元素 */

        /* 邊框:暗色透明 */
        --line-faint:   rgba(26,22,18,0.06);
        --line:         rgba(26,22,18,0.12);
        --line-strong:  rgba(26,22,18,0.22);

        /* 文字 — 深色於淺底,提升對比 */
        --ink:          #1c1814;        /* 主文字(深咖啡)*/
        --ink-soft:     #3a342a;        /* 次要文字 */
        --ink-mute:     #847e6f;        /* 標籤 */
        --ink-faint:    #b3ad9d;        /* 極弱 */

        /* 香檳金(於淺底上需較深以保持可讀)*/
        --gold:         #a67632;        /* 主用(深暗金)*/
        --gold-bright:  #b8854a;        /* 高亮 */
        --gold-deep:    #74521f;        /* 極深 */
        --gold-tint:    rgba(166,118,50,0.10);

        /* 警示 */
        --crimson:      #b03a30;
        --crimson-tint: rgba(176,58,48,0.10);
        --sage:         #5d8a3c;        /* 深草綠 */
        --sage-tint:    rgba(93,138,60,0.12);

        /* Spacing */
        --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px;
        --sp-5: 24px; --sp-6: 32px; --sp-7: 48px; --sp-8: 64px; --sp-9: 96px;

        /* Typography — 與海報一致 */
        --font-sans:  "Noto Sans TC", -apple-system, "PingFang TC",
                      "Helvetica Neue", sans-serif;
        --font-serif: "Noto Serif TC", "Songti TC", Georgia, serif;
        --font-mono:  "JetBrains Mono", "SF Mono", ui-monospace, monospace;
    }

    /* ──────────────── 基礎 ──────────────── */
    html, body, [class*="css"], .stApp {
        background: var(--bg-1) !important;
        color: var(--ink) !important;
        font-family: var(--font-sans);
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }
    .stApp {
        background:
            radial-gradient(1200px 600px at 85% -10%, rgba(166,118,50,.08) 0%, transparent 60%),
            radial-gradient(900px 500px at -10% 110%, rgba(176,58,48,.05) 0%, transparent 60%),
            var(--bg-1) !important;
    }

    .block-container {
        padding-top: var(--sp-6) !important;
        padding-bottom: var(--sp-9) !important;
        max-width: 1380px;
    }

    /* Typography 階層 */
    h1, h2, h3, h4 {
        font-family: var(--font-sans) !important;
        color: var(--ink) !important;
        letter-spacing: -0.02em;
        font-weight: 700 !important;
    }
    body, p, div, span, label {
        font-size: 15px;
        line-height: 1.55;
    }
    .label {
        font-family: var(--font-sans);
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--ink-mute);
    }

    code, kbd {
        font-family: var(--font-mono) !important;
        font-size: 0.85em;
        background: var(--bg-3) !important;
        color: var(--gold) !important;
        border: none !important;
        border-radius: 3px;
        padding: 2px 7px;
    }

    /* ──────────────── Sidebar ──────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-0) !important;
        border-right: 1px solid var(--line-faint) !important;
    }
    section[data-testid="stSidebar"] > div {
        background: var(--bg-0) !important;
        padding-top: var(--sp-5) !important;
    }

    .brand {
        margin-bottom: var(--sp-6);
        padding-bottom: var(--sp-4);
        border-bottom: 1px solid var(--line-faint);
    }
    .brand-mark {
        font-family: var(--font-serif);
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--gold);
        letter-spacing: -0.02em;
        line-height: 1;
        margin-bottom: 4px;
    }
    .brand-mark .dot { color: var(--crimson); }
    .brand-meta {
        font-family: var(--font-sans);
        font-size: 0.78rem;
        color: var(--ink-mute);
        font-weight: 500;
    }

    .nav-label {
        font-family: var(--font-sans);
        font-size: 0.78rem;
        color: var(--ink-mute);
        font-weight: 600;
        margin: var(--sp-5) 0 var(--sp-2) 0;
    }

    /* 輸入框 */
    .stTextInput input {
        background: var(--bg-2) !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 3px !important;
        font-family: var(--font-mono) !important;
        font-size: 0.82rem !important;
        padding: 10px 12px !important;
        transition: border-color .15s ease;
    }
    .stTextInput input:focus {
        border-color: var(--gold) !important;
        box-shadow: none !important;
    }

    /* 主動作按鈕 — 香檳金,中性精緻 */
    div[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: var(--gold) !important;
        color: #1a1612 !important;
        border: none !important;
        border-radius: 3px !important;
        padding: 12px 16px !important;
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.04em !important;
        width: 100%;
        box-shadow: 0 1px 0 rgba(255,255,255,.10) inset,
                    0 8px 24px -6px rgba(212,165,116,.45);
        transition: transform .15s ease, background .15s ease;
    }
    div[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover:not(:disabled) {
        background: var(--gold-bright) !important;
        transform: translateY(-1px);
    }
    div[data-testid="stSidebar"] div.stButton > button[kind="primary"]:disabled {
        background: rgba(212,165,116,.18) !important;
        color: var(--ink-faint) !important;
        box-shadow: none;
        opacity: 1 !important;
    }

    /* Sidebar 次要按鈕:無底色幽靈感 */
    div[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]) {
        background: transparent !important;
        color: var(--ink-soft) !important;
        border: 1px solid var(--line) !important;
        border-radius: 3px !important;
        font-family: var(--font-sans) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        transition: all .15s ease;
    }
    div[data-testid="stSidebar"] div.stButton > button:not([kind="primary"]):hover {
        border-color: var(--gold) !important;
        color: var(--gold) !important;
        background: var(--gold-tint) !important;
    }

    /* 危險按鈕 */
    .btn-danger div.stButton > button {
        background: var(--crimson) !important;
        color: #fbf8f0 !important;
        border: none !important;
        border-radius: 3px !important;
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.04em !important;
        padding: 12px 16px !important;
        width: 100%;
        box-shadow: 0 8px 24px -6px rgba(196,69,58,.45);
        transition: all .15s ease;
    }
    .btn-danger div.stButton > button:hover {
        background: #d6594f !important;
        transform: translateY(-1px);
    }

    /* ──────────────── HERO ──────────────── */
    .hero {
        position: relative;
        padding: var(--sp-8) var(--sp-7) var(--sp-7) calc(var(--sp-7) + 4px);
        margin: var(--sp-3) 0 var(--sp-7) 0;
        background: linear-gradient(135deg,
            rgba(212,165,116,.025) 0%, transparent 50%),
            var(--bg-2);
        border: 1px solid var(--line);
        border-radius: 6px;
        overflow: hidden;
    }
    /* 左側細金條 — 替代四角十字準星 */
    .hero::before {
        content: ""; position: absolute;
        left: 0; top: var(--sp-7); bottom: var(--sp-7);
        width: 3px;
        background: linear-gradient(180deg, var(--gold) 0%, var(--gold-deep) 100%);
        border-radius: 0 2px 2px 0;
    }
    /* 右上角極淡放射 */
    .hero::after {
        content: ""; position: absolute;
        top: -120px; right: -120px;
        width: 360px; height: 360px;
        background: radial-gradient(circle, rgba(212,165,116,.10) 0%, transparent 65%);
        pointer-events: none;
    }

    .hero-eyebrow {
        display: inline-block;
        font-size: 0.78rem;
        color: var(--gold);
        font-weight: 600;
        margin-bottom: var(--sp-4);
        padding: 4px 10px;
        background: var(--gold-tint);
        border-radius: 3px;
        position: relative; z-index: 1;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.25;
        color: var(--ink);
        letter-spacing: -0.025em;
        margin: 0 0 var(--sp-4) 0;
        max-width: 760px;
        word-break: keep-all;       /* 中文字不會在詞中間斷行 */
        overflow-wrap: break-word;  /* 但太長還是會 wrap,不會 overflow */
        text-wrap: balance;          /* 自動平衡多行寬度(現代瀏覽器) */
    }
    .hero-title .accent {
        color: var(--gold);
        /* 強制繼承父元素字級與字重,避免 Streamlit markdown 渲染時把 inline span 縮小 */
        font-size: inherit !important;
        font-weight: inherit !important;
        line-height: inherit !important;
        vertical-align: baseline;
    }

    .hero-lead {
        font-size: 1.05rem;
        color: var(--ink-soft);
        line-height: 1.65;
        max-width: 640px;
        margin-bottom: var(--sp-5);
    }
    .hero-lead b {
        color: var(--ink); font-weight: 600;
    }

    .hero-bylines {
        display: flex; gap: var(--sp-7);
        padding-top: var(--sp-5);
        border-top: 1px solid var(--line-faint);
        flex-wrap: wrap;
    }
    .byline-item {
        display: flex; flex-direction: column; gap: 4px;
    }
    .byline-key {
        font-size: 0.75rem;
        color: var(--ink-mute);
        font-weight: 500;
    }
    .byline-val {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--ink);
    }
    .byline-val b { color: var(--gold); font-weight: 700; }

    /* ──────────────── Section heading(編號徽章 + 標題 + 副標 + 底線) ──────────────── */
    .section-head {
        display: flex; align-items: baseline; gap: 14px;
        margin: var(--sp-8) 0 var(--sp-5) 0;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--line);
    }
    .section-head .num {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--gold);
        letter-spacing: 0.18em;
        padding: 3px 9px;
        background: var(--gold-tint);
        border-radius: 3px;
        flex-shrink: 0;
    }
    .section-head .title {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.01em;
    }
    .section-head .sub-en {
        font-size: 0.82rem;
        color: var(--ink-mute);
        font-weight: 500;
        font-family: var(--font-sans);
    }

    /* ──────────────── Workflow ──────────────── */
    .workflow {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-radius: 6px;
        overflow: hidden;
    }
    .workflow-step {
        padding: var(--sp-6) var(--sp-5);
        position: relative;
        transition: background .2s ease;
    }
    .workflow-step + .workflow-step::before {
        content: ""; position: absolute;
        left: 0; top: var(--sp-5); bottom: var(--sp-5);
        width: 1px; background: var(--line-faint);
    }
    .workflow-step:hover {
        background: linear-gradient(180deg,
            rgba(212,165,116,.04) 0%, transparent 100%);
    }
    .wf-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 28px; height: 28px;
        font-size: 0.85rem; font-weight: 700;
        color: var(--gold);
        background: var(--gold-tint);
        border-radius: 50%;
        margin-bottom: var(--sp-3);
    }
    .wf-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: var(--sp-2);
        letter-spacing: -0.005em;
    }
    .wf-desc {
        font-size: 0.92rem;
        color: var(--ink-soft);
        line-height: 1.6;
    }

    /* ──────────────── Class cards(頂色條 + 圓圖示徽章) ──────────────── */
    .class-card {
        position: relative;
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 26px 24px 24px 24px;
        height: 100%;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(28,24,20,0.04);
        transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    }
    /* 頂部色條 — 4px 厚,色彩 identity 明顯 */
    .class-card::before {
        content: ""; position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: var(--accent);
    }
    .class-card:hover {
        transform: translateY(-3px);
        border-color: var(--line-strong);
        box-shadow: 0 12px 28px -8px rgba(28,24,20,0.12);
    }
    /* 圓形圖示徽章 */
    .class-card .icon-wrap {
        display: inline-flex;
        align-items: center; justify-content: center;
        width: 52px; height: 52px;
        border-radius: 50%;
        background: var(--accent-tint);
        color: var(--accent);
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 18px;
        border: 1.5px solid var(--accent);
    }
    .class-card .tag {
        display: inline-block;
        font-family: var(--font-mono);
        font-size: 0.68rem; font-weight: 700;
        color: var(--accent);
        background: var(--accent-tint);
        padding: 3px 9px;
        border-radius: 3px;
        letter-spacing: 0.14em;
        margin-bottom: 8px;
    }
    .class-card .name {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 8px;
        letter-spacing: -0.01em;
    }
    .class-card .desc {
        font-size: 0.92rem; color: var(--ink-soft);
        line-height: 1.6;
    }

    /* ──────────────── Stat cards ──────────────── */
    .stat-row {
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: var(--sp-3);
        margin: var(--sp-2) 0 var(--sp-6) 0;
    }
    .stat-card {
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: var(--sp-5);
        position: relative;
        transition: transform .15s ease, border-color .15s ease;
    }
    .stat-card::before {
        content: ""; position: absolute;
        left: 0; top: 0; bottom: 0; width: 2px;
        background: var(--accent);
        opacity: 0.7;
        border-radius: 6px 0 0 6px;
    }
    .stat-card:hover {
        border-color: var(--line-strong);
        transform: translateY(-1px);
    }
    .stat-card .stat-label {
        font-size: 0.82rem;
        color: var(--ink-mute);
        font-weight: 500;
        margin-bottom: var(--sp-3);
    }
    .stat-card .stat-value {
        font-size: 2.4rem;
        font-weight: 800;
        color: var(--accent);
        line-height: 1;
        letter-spacing: -0.03em;
        font-variant-numeric: tabular-nums;
    }
    .stat-total { --accent: var(--ink); }
    .stat-bad   { --accent: var(--crimson); }
    .stat-good  { --accent: var(--sage); }
    .stat-nf    { --accent: var(--ink-mute); }

    /* ──────────────── Folder breadcrumb ──────────────── */
    .folder-bar {
        display: inline-flex; align-items: center; gap: var(--sp-3);
        padding: 8px var(--sp-4);
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-radius: 4px;
        margin-bottom: var(--sp-3);
        font-size: 0.88rem;
        color: var(--ink-soft);
    }
    .folder-bar .key {
        color: var(--gold);
        font-size: 0.78rem;
        font-weight: 600;
        padding-right: var(--sp-3);
        border-right: 1px solid var(--line);
    }
    .folder-bar .path {
        font-family: var(--font-mono);
        font-size: 0.82rem;
        color: var(--ink);
    }

    /* ──────────────── Tabs(滑動 underline) ──────────────── */
    div[data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--line) !important;
        border-top: none !important;
        gap: var(--sp-2);
        padding: 0 !important;
        margin-bottom: var(--sp-4);
        position: relative;
    }
    button[data-baseweb="tab"] {
        background: transparent !important;
        color: var(--ink-mute) !important;
        font-family: var(--font-sans) !important;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.005em !important;
        padding: var(--sp-3) var(--sp-5) !important;
        border-radius: 0 !important;
        border: none !important;
        transition: color .25s ease;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--ink) !important;
        background: transparent !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }
    button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        color: var(--ink-soft) !important;
    }
    /* BaseWeb 內建的滑動高亮元素 — 把它變成金色 underline */
    div[data-baseweb="tab-highlight"] {
        display: block !important;
        background: var(--gold) !important;
        height: 2px !important;
        bottom: 0 !important;
        border-radius: 1px;
        box-shadow: 0 0 8px rgba(212,165,116,.5);
        transition: left .35s cubic-bezier(0.4, 0, 0.2, 1),
                    width .35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-baseweb="tab-border"] { display: none !important; }

    /* Tab 內容區滑入動畫 */
    /* 動畫拿掉 — opacity: 0 的瞬間造成「白閃」,而且每次 fragment 重渲染都觸發 */

    /* ──────────────── 底片膠卷分隔線 ──────────────── */
    .film-strip {
        position: relative;
        height: 44px;
        margin: var(--sp-7) calc(-1 * var(--sp-3));
        background:
            linear-gradient(90deg,
                transparent 0,
                var(--bg-3) 40px,
                var(--bg-3) calc(100% - 40px),
                transparent 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    /* 上下兩排齒孔(sprocket holes) */
    .film-strip::before,
    .film-strip::after {
        content: "";
        position: absolute;
        left: 40px; right: 40px;
        height: 9px;
        background: repeating-linear-gradient(
            to right,
            transparent 0,
            transparent 7px,
            var(--bg-1) 7px,
            var(--bg-1) 17px
        );
    }
    .film-strip::before { top: 5px; }
    .film-strip::after  { bottom: 5px; }

    /* 中間的「曝光區」訊息文字 */
    .film-strip .film-text {
        position: relative; z-index: 1;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--gold);
        letter-spacing: 0.08em;
        padding: 0 var(--sp-5);
        background: var(--bg-3);
    }
    .film-strip .film-text::before,
    .film-strip .film-text::after {
        content: "•";
        margin: 0 var(--sp-3);
        color: var(--ink-faint);
        letter-spacing: 0;
    }

    /* 變體:小尺寸 film strip(分頁內用) */
    .film-strip.mini {
        height: 28px;
        margin: var(--sp-5) calc(-1 * var(--sp-3));
    }
    .film-strip.mini::before,
    .film-strip.mini::after {
        left: 24px; right: 24px;
        height: 6px;
        background: repeating-linear-gradient(
            to right,
            transparent 0, transparent 5px,
            var(--bg-1) 5px, var(--bg-1) 12px
        );
    }
    .film-strip.mini::before { top: 3px; }
    .film-strip.mini::after  { bottom: 3px; }
    .film-strip.mini .film-text { font-size: 0.55rem; letter-spacing: 0.35em; }

    /* ──────────────── 照片 ──────────────── */
    [data-testid="stImage"] img {
        border-radius: 3px;
        transition: opacity .2s ease, transform .2s ease;
    }
    [data-testid="stImage"]:hover img {
        transform: scale(1.005);
    }

    /* ──────────────── Bad card 容器 ──────────────── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-2) !important;
        border: 1px solid var(--line) !important;
        border-radius: 5px !important;
        padding: var(--sp-4) !important;
        transition: border-color .15s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: var(--line-strong) !important;
    }

    /* ──────────────── 信心徽章 ──────────────── */
    .conf {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 3px 10px;
        font-size: 0.82rem; font-weight: 600;
        border-radius: 3px;
        font-variant-numeric: tabular-nums;
    }
    .conf-bad  {
        background: var(--crimson-tint);
        color: var(--crimson);
        border: 1px solid rgba(176,58,48,0.32);
    }
    .conf-good {
        background: var(--sage-tint);
        color: var(--sage);
        border: 1px solid rgba(93,138,60,0.32);
    }

    /* ──────────────── 檔名 ──────────────── */
    .filename {
        font-family: var(--font-mono);
        font-size: 0.82rem;
        color: var(--ink-soft);
        margin-top: var(--sp-2);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .img-cap {
        font-size: 0.72rem; font-weight: 500;
        color: var(--ink-mute);
        text-align: center;
        margin-top: 4px;
    }

    /* ──────────────── Empty state ──────────────── */
    .empty {
        text-align: center;
        padding: var(--sp-9) var(--sp-5);
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-radius: 6px;
    }
    .empty-glyph {
        font-size: 3.5rem;
        color: var(--gold);
        opacity: 0.6;
        line-height: 1;
        margin-bottom: var(--sp-4);
    }
    .empty-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: var(--sp-3);
        letter-spacing: -0.02em;
    }
    .empty-desc {
        color: var(--ink-soft);
        font-size: 0.98rem;
        max-width: 460px;
        margin: 0 auto;
        line-height: 1.65;
    }

    /* ──────────────── Action bar(刪除前的確認區) ──────────────── */
    .action-bar {
        display: flex; align-items: center; justify-content: space-between;
        gap: var(--sp-5);
        padding: var(--sp-5);
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-left: 2px solid var(--crimson);
        border-radius: 4px;
        margin-top: var(--sp-5);
    }
    .action-bar .count {
        display: flex; align-items: baseline; gap: var(--sp-3);
    }
    .action-bar .count .num {
        font-size: 1.8rem; font-weight: 800;
        color: var(--crimson); line-height: 1;
        letter-spacing: -0.03em;
        font-variant-numeric: tabular-nums;
    }
    .action-bar .count .lbl {
        font-size: 0.88rem; font-weight: 500;
        color: var(--ink-soft);
    }

    /* ──────────────── Scan card(置中大進度) ──────────────── */
    .scan-card {
        margin: 80px auto 60px;
        max-width: 620px;
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 56px 64px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .scan-card::before {
        content: ""; position: absolute;
        top: -160px; left: 50%; transform: translateX(-50%);
        width: 420px; height: 420px;
        background: radial-gradient(circle, rgba(212,165,116,.15) 0%, transparent 60%);
        pointer-events: none;
        animation: bg-pulse 3s ease-in-out infinite;
    }
    @keyframes bg-pulse {
        0%, 100% { opacity: 0.6; transform: translateX(-50%) scale(1); }
        50%      { opacity: 1.0; transform: translateX(-50%) scale(1.08); }
    }

    .scan-label {
        font-size: 0.9rem;
        color: var(--gold);
        font-weight: 600;
        margin-bottom: 36px;
        position: relative; z-index: 1;
    }
    .scan-label .pulse-dot {
        display: inline-block; margin-right: 10px;
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--gold);
        box-shadow: 0 0 8px var(--gold);
        animation: dot-pulse 1.2s ease-in-out infinite;
        vertical-align: middle;
    }
    @keyframes dot-pulse {
        0%, 100% { opacity: 1;   transform: scale(1); }
        50%      { opacity: 0.4; transform: scale(0.85); }
    }

    .scan-filename-wrap {
        margin-bottom: 32px; position: relative; z-index: 1;
    }
    .scan-filename {
        font-family: var(--font-mono);
        font-size: 1.05rem;
        color: var(--ink);
        padding: 14px 24px;
        background: var(--bg-3);
        border-radius: 3px;
        border-left: 2px solid var(--gold);
        display: inline-block;
        min-width: 320px;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        letter-spacing: 0.02em;
    }

    .scan-progress {
        width: 100%; height: 4px;
        background: var(--bg-3);
        border-radius: 2px;
        overflow: hidden;
        margin-bottom: 36px;
        position: relative;
    }
    .scan-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--gold-deep), var(--gold));
        transition: width 0.25s ease-out;
        box-shadow: 0 0 14px rgba(212,165,116,.7);
    }

    .scan-stats {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 24px;
        padding-top: 28px;
        border-top: 1px solid var(--line-faint);
        position: relative; z-index: 1;
    }
    .scan-stat-num {
        font-size: 1.9rem;
        font-weight: 800;
        color: var(--ink);
        line-height: 1;
        margin-bottom: 6px;
        letter-spacing: -0.03em;
        font-variant-numeric: tabular-nums;
    }
    .scan-stat-num.accent { color: var(--gold); }
    .scan-stat-lbl {
        font-size: 0.78rem;
        color: var(--ink-mute);
        font-weight: 500;
    }

    /* ──────────────── Progress(sidebar fallback) ──────────────── */
    div[data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, var(--gold-deep), var(--gold)) !important;
        height: 3px;
    }
    div[data-testid="stProgress"] > div > div {
        height: 3px; background: var(--bg-3);
    }
    div[data-testid="stProgress"] p {
        font-size: 0.85rem !important;
        color: var(--ink-mute) !important;
    }

    /* ──────────────── Expander / Alert ──────────────── */
    div[data-testid="stExpander"] {
        background: var(--bg-2);
        border: 1px solid var(--line) !important;
        border-radius: 4px;
    }
    div[data-testid="stExpander"] summary {
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--ink-soft) !important;
        padding: var(--sp-3) var(--sp-4) !important;
    }
    div[data-testid="stExpander"] summary:hover { color: var(--gold) !important; }

    div[data-testid="stAlert"] {
        background: var(--bg-2) !important;
        border: 1px solid var(--line) !important;
        border-left: 2px solid var(--gold) !important;
        border-radius: 4px !important;
    }

    /* Checkbox */
    div[data-testid="stCheckbox"] {
        margin-top: var(--sp-3);
        margin-bottom: var(--sp-2);
    }
    div[data-testid="stCheckbox"] label {
        font-size: 0.88rem;
        color: var(--ink-soft);
    }

    /* Recent folder */
    .recent-row {
        font-family: var(--font-mono);
        font-size: 0.8rem;
        color: var(--ink-soft);
        line-height: 1.8;
    }

    /* Footer */
    .footer {
        margin-top: var(--sp-8);
        padding-top: var(--sp-5);
        border-top: 1px solid var(--line-faint);
        text-align: center;
        font-size: 0.8rem;
        color: var(--ink-mute);
    }
    .footer .accent { color: var(--gold); margin: 0 8px; }

    /* 隱藏 Streamlit 預設 */
    #MainMenu, footer:not(.footer) { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; }
    hr { border-color: var(--line) !important; }

    /* Spinner 文字 */
    [data-testid="stSpinner"] > div > span {
        font-size: 0.92rem !important;
        color: var(--ink-soft) !important;
    }
    </style>
    """, unsafe_allow_html=True)


inject_theme()


# ──────────────────────────────────────────────────────────────────────────
# 3. Session State
# ──────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "scan_results": None,
    "select_all": False,
    "error_logs": [],
    "current_folder": None,
    "view_mode": None,  # None / "trash" — 切換主畫面顯示
}
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ──────────────────────────────────────────────────────────────────────────
# 4. 工具函式
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_thumbnail(img_path, mtime, max_size=(200, 200)):
    try:
        img = Image.open(img_path)
        img.thumbnail(max_size)
        return img
    except Exception:
        return None


def toggle_select_all():
    st.session_state.select_all = not st.session_state.select_all
    if st.session_state.scan_results:
        for item in st.session_state.scan_results["Bad"]:
            st.session_state[f"del_{item['path']}"] = st.session_state.select_all


@st.fragment
def _render_bad_card(item: dict) -> None:
    """渲染單張 Bad 卡片 — 用 @st.fragment 隔離,checkbox 點擊只重渲染這張卡,
    不會觸發整個 page 的 rerun(原本 60 張卡 + 120 張圖一起重畫,非常慢)。"""
    fname = item["path"].name
    pct = item["prob"] * 100
    chk_key = f"del_{item['path']}"
    if chk_key not in st.session_state:
        st.session_state[chk_key] = False

    with st.container(border=True):
        st.markdown(
            f"<span class='conf conf-bad'>崩壞機率 {pct:.1f}%</span>",
            unsafe_allow_html=True,
        )
        st.checkbox("勾選刪除", key=chk_key, label_visibility="visible")

        img_col1, img_col2 = st.columns(2, gap="small")
        with img_col1:
            thumb = load_thumbnail(item["path"], item["mtime"])
            if thumb:
                st.image(thumb)
            st.markdown(
                "<div class='img-cap'>原圖</div>", unsafe_allow_html=True,
            )
        with img_col2:
            st.image(item["face_img"])
            st.markdown(
                "<div class='img-cap'>AI 鎖定區</div>", unsafe_allow_html=True,
            )
        st.markdown(
            f"<div class='filename' title='{html.escape(fname)}'>"
            f"{html.escape(fname)}</div>",
            unsafe_allow_html=True,
        )


def _paginate(items: list, key_prefix: str, default_size: int = 30) -> tuple[list, int, int, int]:
    """渲染分頁控制列,回傳 (本頁的 items, 目前頁碼, 總頁數, 總數)。

    分頁可以避免 Streamlit 把 500+ 張卡片一次全部渲染,大幅改善切換 tab 的卡頓。
    """
    if not items:
        return items, 1, 1, 0
    total = len(items)
    size_key = f"{key_prefix}_pg_size"
    page_key = f"{key_prefix}_pg_num"
    st.session_state.setdefault(size_key, default_size)
    st.session_state.setdefault(page_key, 1)

    size = st.session_state[size_key]
    n_pages = max(1, (total + size - 1) // size)
    page = max(1, min(st.session_state[page_key], n_pages))
    st.session_state[page_key] = page

    options = [30, 60, 120, total]
    options = sorted(set(options))  # 去重 + 排序

    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 3.6])
    with c1:
        new_size = st.selectbox(
            "每頁顯示", options,
            index=options.index(size) if size in options else 1,
            key=f"{key_prefix}_size_sel",
            label_visibility="collapsed",
        )
        if new_size != size:
            st.session_state[size_key] = new_size
            st.session_state[page_key] = 1
            st.rerun()
    with c2:
        if st.button("← 上一頁", disabled=page <= 1,
                     key=f"{key_prefix}_prev_btn", use_container_width=True):
            st.session_state[page_key] = page - 1
            st.rerun()
    with c3:
        if st.button("下一頁 →", disabled=page >= n_pages,
                     key=f"{key_prefix}_next_btn", use_container_width=True):
            st.session_state[page_key] = page + 1
            st.rerun()
    with c4:
        st.markdown(
            f"<div style='padding-top:8px;color:#847e6f;font-size:.88rem;'>"
            f"第 <b style='color:#1c1814;'>{page}</b> / {n_pages} 頁  ·  "
            f"共 <b style='color:#1c1814;'>{total}</b> 張</div>",
            unsafe_allow_html=True,
        )

    start = (page - 1) * size
    end = start + size
    return items[start:end], page, n_pages, total


def load_recent_folders() -> list[str]:
    if not RECENT_FOLDERS_FILE.exists():
        return []
    try:
        return json.loads(RECENT_FOLDERS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_recent_folder(folder: str) -> None:
    recent = load_recent_folders()
    if folder in recent:
        recent.remove(folder)
    recent.insert(0, folder)
    recent = recent[:RECENT_FOLDERS_MAX]
    RECENT_FOLDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RECENT_FOLDERS_FILE.write_text(
        json.dumps(recent, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def list_scannable_images(folder: Path) -> list[Path]:
    trash_name = "Trash"
    out: list[Path] = []
    for f in folder.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in VALID_EXTS:
            continue
        try:
            rel_parts = f.relative_to(folder).parts
            if rel_parts and rel_parts[0] == trash_name:
                continue
        except ValueError:
            pass
        out.append(f)
    return out


def _format_eta(seconds: float) -> str:
    if seconds < 1:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    m = int(seconds // 60); s = int(seconds % 60)
    return f"{m}m {s:02d}s"


def _render_scan_card(placeholder, *, fname: str, i: int, n: int, eta_str: str) -> None:
    """把進度卡片渲染到指定 placeholder(主畫面置中)。"""
    pct = (i / n) * 100 if n else 0
    placeholder.markdown(f"""
    <div class="scan-card">
        <div class="scan-label">
            <span class="pulse-dot"></span>神經網路正在分析…
        </div>
        <div class="scan-filename-wrap">
            <div class="scan-filename">{html.escape(fname)}</div>
        </div>
        <div class="scan-progress">
            <div class="scan-progress-fill" style="width:{pct:.2f}%;"></div>
        </div>
        <div class="scan-stats">
            <div>
                <div class="scan-stat-num">{i}</div>
                <div class="scan-stat-lbl">共 {n} 張</div>
            </div>
            <div>
                <div class="scan-stat-num accent">{pct:.1f}%</div>
                <div class="scan-stat-lbl">完成度</div>
            </div>
            <div>
                <div class="scan-stat-num">{html.escape(eta_str)}</div>
                <div class="scan-stat-lbl">預估剩餘</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def run_scan(folder_path: Path, placeholder) -> dict:
    """執行批次推論。進度卡片渲染到傳入的 placeholder(預期是 main area 的 st.empty())。"""
    image_files = list_scannable_images(folder_path)
    if not image_files:
        placeholder.empty()
        st.sidebar.warning("No images found.")
        return {"Good": [], "Bad": [], "NoFace": []}

    st.session_state.error_logs = []
    results = {"Good": [], "Bad": [], "NoFace": []}
    n = len(image_files)
    t_start = time.time()

    # 顯示初始載入畫面
    _render_scan_card(placeholder, fname="正在載入神經網路…",
                      i=0, n=n, eta_str="—")

    for i, img_path in enumerate(image_files, 1):
        elapsed = time.time() - t_start
        rate = i / max(elapsed, 0.01)
        eta = (n - i) / rate if rate > 0 else 0
        _render_scan_card(
            placeholder, fname=img_path.name,
            i=i, n=n, eta_str=_format_eta(eta),
        )

        try:
            res = predict_image_quality(str(img_path))
            item = {
                "path": img_path, "mtime": img_path.stat().st_mtime,
                "prob": res["probability"], "label": res["label"],
                "probs": res["probs"],
            }
            if res["label"] == "Bad":
                item["face_img"] = res["face_image"]
            results[res["label"]].append(item)
        except NoFaceDetectedError:
            results["NoFace"].append({
                "path": img_path, "mtime": img_path.stat().st_mtime,
            })
        except Exception as e:
            st.session_state.error_logs.append(f"{img_path.name}: {e}")

    placeholder.empty()
    results["Bad"].sort(key=lambda x: x["prob"], reverse=True)
    results["Good"].sort(key=lambda x: x["prob"], reverse=True)
    return results


def restore_from_trash(trash_dir: Path) -> tuple[int, list[str]]:
    """還原 Trash 中所有照片(全部)。"""
    manifest_file = trash_dir / "manifest.json"
    if not manifest_file.exists():
        return 0, []
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception:
        return 0, []
    return _restore_entries(manifest, manifest_file, set())  # 空集合 = 全部還原


def restore_selected_from_trash(
    trash_dir: Path,
    selected_trash_paths: set[str],
) -> tuple[int, list[str]]:
    """還原指定的照片(只還原 trash_path 在 selected_trash_paths 內的)。"""
    manifest_file = trash_dir / "manifest.json"
    if not manifest_file.exists():
        return 0, []
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception:
        return 0, []
    return _restore_entries(manifest, manifest_file, selected_trash_paths)


def _restore_entries(
    manifest: list[dict],
    manifest_file: Path,
    selected: set[str],
) -> tuple[int, list[str]]:
    """共用邏輯。selected 空集合代表「全部還原」,否則只還原指定的。"""
    restored = 0
    failed: list[str] = []
    remaining: list[dict] = []
    for entry in manifest:
        # 不在選取清單中 → 保留在 trash
        if selected and entry["trash_path"] not in selected:
            remaining.append(entry)
            continue

        src = Path(entry["trash_path"])
        dst = Path(entry["original_path"])
        if not src.exists():
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

    manifest_file.write_text(
        json.dumps(remaining, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return restored, failed


def _render_trash_view() -> None:
    """主畫面:Trash 內容檢視 + 選擇性還原。"""
    if not st.session_state.current_folder:
        st.warning("尚未指定資料夾,無法查看 Trash。")
        return

    trash_dir = Path(st.session_state.current_folder) / "Trash"
    manifest_file = trash_dir / "manifest.json"

    # 頂部:返回按鈕 + 標題 + 全部還原
    head_cols = st.columns([1, 4, 1.5])
    with head_cols[0]:
        if st.button("← 返回掃描", key="back_from_trash", use_container_width=True):
            st.session_state.view_mode = None
            st.rerun()

    if not manifest_file.exists():
        st.info("Trash 是空的。")
        return
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception:
        manifest = []

    with head_cols[1]:
        st.markdown(
            f"<h2 style='margin:0;padding-top:4px;'>♻ Trash · {len(manifest)} 張照片</h2>",
            unsafe_allow_html=True,
        )
    with head_cols[2]:
        if st.button(f"還原全部 ({len(manifest)})", key="restore_all_full",
                     use_container_width=True):
            with st.spinner("還原中..."):
                n_ok, fails = restore_from_trash(trash_dir)
            if n_ok:
                st.success(f"✅ 已還原 {n_ok} 張")
            if fails:
                st.error(f"❌ {len(fails)} 張失敗")
            st.rerun()

    if not manifest:
        st.info("Trash 是空的。")
        return

    # 選擇性還原控制列
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='padding:6px 0;color:#847e6f;font-size:.9rem;'>"
        "勾選你想還原的照片,按下方按鈕復原(沒勾選的會繼續留在 Trash)。"
        "</div>",
        unsafe_allow_html=True,
    )

    # 即時抓取目前所有勾選
    selected_paths = {
        entry["trash_path"]
        for entry in manifest
        if st.session_state.get(f"trash_sel_{entry['trash_path']}", False)
    }
    n_sel = len(selected_paths)

    # 還原選取按鈕
    if st.button(
        f"⤺  還原勾選的 {n_sel} 張" if n_sel else "⤺  還原勾選的(先勾選)",
        key="restore_selected_btn",
        type="primary",
        disabled=n_sel == 0,
        use_container_width=True,
    ):
        with st.spinner("還原中..."):
            n_ok, fails = restore_selected_from_trash(trash_dir, selected_paths)
        if n_ok:
            st.success(f"✅ 已還原 {n_ok} 張到原路徑")
        if fails:
            st.error(f"❌ {len(fails)} 張失敗")
        # 清掉 checkbox 狀態
        for p in selected_paths:
            st.session_state.pop(f"trash_sel_{p}", None)
        st.rerun()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Trash 卡片網格(4 欄)
    cols = st.columns(4, gap="small")
    for i, entry in enumerate(manifest):
        trash_path = Path(entry["trash_path"])
        orig_path = Path(entry["original_path"])
        with cols[i % 4]:
            with st.container(border=True):
                sel_key = f"trash_sel_{entry['trash_path']}"
                st.checkbox("勾選還原", key=sel_key)

                if trash_path.exists():
                    try:
                        mtime = trash_path.stat().st_mtime
                        thumb = load_thumbnail(trash_path, mtime)
                        if thumb:
                            st.image(thumb, use_container_width=True)
                    except Exception:
                        st.warning("無法顯示縮圖")
                else:
                    st.warning("⚠ 檔案已不存在")

                st.markdown(
                    f"<div class='filename' title='{html.escape(orig_path.name)}'>"
                    f"{html.escape(orig_path.name)}</div>",
                    unsafe_allow_html=True,
                )
                deleted_at = entry.get("deleted_at", "")[:16].replace("T", " ")
                st.caption(f"刪除於 {deleted_at}")


# ──────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-mark">Darkroom<span class="dot">.</span></div>
        <div class="brand-meta">AI 表情相簿管家</div>
    </div>
    """, unsafe_allow_html=True)

    model_exists = MODEL_PATH.exists()
    if not model_exists:
        st.error("Model not found.")
        st.markdown(
            "**Download:**\n"
            "```bash\n"
            "curl -L https://github.com/Hunter20041004/smart-album-cleaner/"
            "releases/latest/download/mobilenet_face.pth \\\n"
            "    -o models/mobilenet_face.pth\n"
            "```"
        )

    st.markdown(
        "<div class='nav-label'>目標資料夾</div>",
        unsafe_allow_html=True,
    )
    target_folder = st.text_input(
        "Source folder path", label_visibility="collapsed",
        value=st.session_state.get("current_folder") or "./datasets/raw",
        key="folder_input",
    )

    recent = load_recent_folders()
    if recent:
        with st.expander("最近用過", expanded=False):
            for rf in recent:
                cols = st.columns([5, 1])
                with cols[0]:
                    disp = rf if len(rf) < 28 else "…" + rf[-25:]
                    st.markdown(
                        f"<div class='recent-row'>{html.escape(disp)}</div>",
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    if st.button("→", key=f"recent_{rf}"):
                        st.session_state.current_folder = rf
                        st.rerun()

    st.markdown(
        "<div class='nav-label'>啟動</div>",
        unsafe_allow_html=True,
    )
    scan_clicked = st.button(
        "🚀  啟動 AI 掃描", type="primary",
        disabled=not model_exists, key="scan_button",
    )

    if scan_clicked:
        folder_path = Path(target_folder)
        if not folder_path.exists() or not folder_path.is_dir():
            st.error("Folder not found.")
        else:
            # 只設旗標,實際掃描在 main area 渲染進度卡
            save_recent_folder(str(folder_path))
            st.session_state.current_folder = str(folder_path)
            st.session_state.pending_scan = str(folder_path)
            st.session_state.select_all = False
            st.session_state.scan_results = None

    if st.session_state.scan_results:
        if st.button("清空結果", key="clear_btn"):
            st.session_state.scan_results = None
            st.session_state.error_logs = []
            st.rerun()

    # Trash restore
    if st.session_state.current_folder:
        trash_dir = Path(st.session_state.current_folder) / "Trash"
        manifest_path = trash_dir / "manifest.json"
        if manifest_path.exists():
            try:
                n_in_trash = len(json.loads(manifest_path.read_text(encoding="utf-8")))
            except Exception:
                n_in_trash = 0
            if n_in_trash > 0:
                with st.expander(f"♻ Trash · {n_in_trash} 張", expanded=False):
                    st.caption("查看詳細內容 + 選擇性還原")
                    if st.button(
                        f"📁  查看 Trash 內容",
                        key="view_trash_btn",
                        use_container_width=True,
                        type="primary",
                    ):
                        st.session_state.view_mode = "trash"
                        st.rerun()
                    st.caption("或直接全部還原(快速):")
                    if st.button(
                        f"⤺  還原全部 {n_in_trash} 張",
                        key="restore_all_btn",
                        use_container_width=True,
                    ):
                        with st.spinner("還原中..."):
                            n_ok, fails = restore_from_trash(trash_dir)
                        if n_ok: st.success(f"✅ 已還原 {n_ok} 張")
                        if fails: st.error(f"❌ {len(fails)} 張失敗")
                        st.rerun()

    if st.session_state.error_logs:
        with st.expander(f"⚠ 錯誤紀錄 · {len(st.session_state.error_logs)}"):
            for err in st.session_state.error_logs:
                st.caption(err)

    st.markdown(
        "<div class='footer' style='margin-top:30px;padding-top:14px;'>"
        "100% 本機運行 <span class='accent'>·</span> MIT License"
        "</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────────────────

# 有 pending_scan → 在主畫面渲染大張進度卡並執行掃描
if st.session_state.get("pending_scan"):
    folder = st.session_state.pop("pending_scan")
    progress_placeholder = st.empty()
    st.session_state.scan_results = run_scan(Path(folder), progress_placeholder)
    st.rerun()

# Trash 檢視模式優先(從 sidebar 「查看 Trash 內容」按鈕進入)
if st.session_state.get("view_mode") == "trash":
    _render_trash_view()
elif st.session_state.scan_results is None:
    # ── Hero ──
    st.markdown(
        '<div class="hero">'
        '<span class="hero-eyebrow">AI 相簿清理工具 · v0.5</span>'
        '<div class="hero-title">'
        '自動挑出相簿裡的<span class="accent">表情崩壞照片</span>,'
        '<br>一鍵軟刪除。'
        '</div>'
        '<div class="hero-lead">'
        '用自訓的 <b>MobileNetV3-Large</b> 神經網路,從你的相簿挑出閉眼、嘴歪、模糊等廢片。'
        '<b>本機運行,完全離線</b> — 你的照片不會離開這台電腦。'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Workflow ──
    st.markdown("""
    <div class="section-head">
        <span class="num">01</span>
        <span class="title">使用流程</span>
        <span class="sub-en">Workflow</span>
    </div>
    <div class="workflow">
        <div class="workflow-step">
            <div class="wf-num">1</div>
            <div class="wf-title">輸入路徑</div>
            <div class="wf-desc">在左側填入相簿資料夾,或從「最近用過」一鍵切換。</div>
        </div>
        <div class="workflow-step">
            <div class="wf-num">2</div>
            <div class="wf-title">啟動掃描</div>
            <div class="wf-desc">點擊 Run AI Scan,神經網路會逐張分析照片表情。</div>
        </div>
        <div class="workflow-step">
            <div class="wf-num">3</div>
            <div class="wf-title">挑出廢片</div>
            <div class="wf-desc">在 Bad 分頁勾選你同意刪除的照片。</div>
        </div>
        <div class="workflow-step">
            <div class="wf-num">4</div>
            <div class="wf-title">安全刪除</div>
            <div class="wf-desc">軟刪除到 Trash,可從 sidebar 隨時還原。</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Class cards ──
    st.markdown("""
    <div class="section-head">
        <span class="num">02</span>
        <span class="title">AI 怎麼分類</span>
        <span class="sub-en">Classification</span>
    </div>
    """, unsafe_allow_html=True)
    cols = st.columns(3, gap="medium")
    cards = [
        ("✓", "GOOD",   "完美表情",     "表情自然、雙眼張開、清晰可見",
         "var(--sage)",     "var(--sage-tint)"),
        ("⚠", "BAD",    "建議刪除",     "閉眼、嘴歪、表情扭曲、明顯模糊",
         "var(--crimson)",  "var(--crimson-tint)"),
        ("○", "NOFACE", "未偵測到人臉", "背影、風景、寵物、或臉太小",
         "var(--ink-mute)", "rgba(132,126,111,0.12)"),
    ]
    for col, (glyph, tag, name, desc, color, tint) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="class-card" style="--accent: {color}; --accent-tint: {tint};">
                <div class="icon-wrap">{glyph}</div>
                <div class="tag">{tag}</div>
                <div class="name">{name}</div>
                <div class="desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("展開技術細節", expanded=False):
        st.markdown("""
        - **人臉偵測**: MediaPipe 雙模型 + letterbox 自動裁切
        - **特徵抽取**: MobileNetV3-Large(ImageNet 預訓 + SE Attention)
        - **訓練策略**: Stage-1 凍結 → Stage-2 解凍最後 3 個 blocks + weight decay
        - **資料增強**: RandomResizedCrop · ColorJitter · RandomErasing
        - **推論優化**: TTA(水平翻轉)+ 自動閾值調校
        - **裝置**: CUDA · MPS(Apple Silicon)· CPU
        """)

    st.markdown("""
    <div class="footer">
        Darkroom <span class="accent">·</span>
        基於 PyTorch + Streamlit 打造 <span class="accent">·</span> MIT License
    </div>
    """, unsafe_allow_html=True)

else:
    # ──────────── 結果頁 ────────────
    results = st.session_state.scan_results
    n_good = len(results["Good"])
    n_bad  = len(results["Bad"])
    n_nf   = len(results["NoFace"])
    n_total = n_good + n_bad + n_nf

    folder_disp = st.session_state.current_folder or ""
    if len(folder_disp) > 60:
        folder_disp = "…" + folder_disp[-57:]
    st.markdown(
        f"<div class='folder-bar'>"
        f"<span class='key'>掃描路徑</span>"
        f"<span class='path'>{html.escape(folder_disp)}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card stat-total">
            <div class="stat-label">總計掃描</div>
            <div class="stat-value">{n_total}</div>
        </div>
        <div class="stat-card stat-bad">
            <div class="stat-label">建議刪除</div>
            <div class="stat-value">{n_bad}</div>
        </div>
        <div class="stat-card stat-good">
            <div class="stat-label">完美表情</div>
            <div class="stat-value">{n_good}</div>
        </div>
        <div class="stat-card stat-nf">
            <div class="stat-label">未偵測到</div>
            <div class="stat-value">{n_nf}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if n_bad + n_good > 0:
        with st.expander("📊 信心分布 — 看模型判斷有多堅定", expanded=False):
            import numpy as np
            for label, data, glyph in [
                ("Good", [it["prob"] for it in results["Good"]], "✓"),
                ("Bad",  [it["prob"] for it in results["Bad"]],  "⚠"),
            ]:
                if not data: continue
                med = np.median(data)
                st.markdown(
                    f"<div style='font-size:.92rem;font-weight:600;"
                    f"color:#a67632;margin:14px 0 8px 0;'>"
                    f"{glyph} {label}  ·  共 {len(data)} 張  ·  中位信心 {med*100:.1f}%</div>",
                    unsafe_allow_html=True,
                )
                bins = np.histogram(data, bins=10, range=(0, 1))[0]
                st.bar_chart({f"{i*10}-{(i+1)*10}%": int(bins[i]) for i in range(10)})

    # 底片膠卷分隔(stats → tabs 之間)
    st.markdown("""
    <div class="film-strip mini">
        <span class="film-text">掃描結果</span>
    </div>
    """, unsafe_allow_html=True)

    tab_bad, tab_good, tab_noface = st.tabs([
        f"⚠  建議刪除 · {n_bad}",
        f"✓  完美表情 · {n_good}",
        f"○  未偵測到人臉 · {n_nf}",
    ])

    # ── Bad ──
    with tab_bad:
        if results["Bad"]:
            ctrl_cols = st.columns([1.4, 3])
            with ctrl_cols[0]:
                st.button("全選 / 全清(全部)",
                          on_click=toggle_select_all,
                          use_container_width=True, key="select_all_btn")
            with ctrl_cols[1]:
                st.markdown(
                    "<div style='padding-top:10px;font-size:.88rem;color:#847f72;'>"
                    "依信心高到低排序</div>",
                    unsafe_allow_html=True,
                )

            # ─── 永遠顯示刪除按鈕(放最上方,使用者勾完不用滾到底就能按)─────
            st.markdown(
                "<div style='padding:6px 0 6px 0;color:#847e6f;font-size:.88rem;'>"
                "勾選下方照片,按下面這顆紅色按鈕移到 Trash。"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='btn-danger'>", unsafe_allow_html=True)
            if st.button(
                "🗑  確認將勾選的照片移到 Trash",
                key="delete_btn",
                use_container_width=True,
            ):
                # 即時抓最新勾選狀態,進行軟刪除
                files_to_delete = [
                    it["path"] for it in results["Bad"]
                    if st.session_state.get(f"del_{it['path']}", False)
                ]
                if not files_to_delete:
                    st.warning("還沒勾選任何照片喔!")
                else:
                    trash_dir = Path(target_folder) / "Trash"
                    trash_dir.mkdir(exist_ok=True)
                    manifest = []
                    moved_paths: set = set()

                    for fp in files_to_delete:
                        if fp.exists():
                            dest = trash_dir / fp.name
                            if dest.exists():
                                dest = trash_dir / (
                                    f"{fp.stem}_"
                                    f"{datetime.now().strftime('%H%M%S')}"
                                    f"{fp.suffix}"
                                )
                            shutil.move(str(fp), str(dest))
                            manifest.append({
                                "original_path": str(fp),
                                "trash_path": str(dest),
                                "deleted_at": datetime.now().isoformat(),
                            })
                            moved_paths.add(fp)

                    # 寫 manifest
                    manifest_file = trash_dir / "manifest.json"
                    existing = (
                        json.loads(manifest_file.read_text(encoding="utf-8"))
                        if manifest_file.exists() else []
                    )
                    existing.extend(manifest)
                    manifest_file.write_text(
                        json.dumps(existing, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )

                    # ★ 只從結果中移除已刪除的項目,保留其他結果讓使用者繼續操作
                    st.session_state.scan_results["Bad"] = [
                        it for it in st.session_state.scan_results["Bad"]
                        if it["path"] not in moved_paths
                    ]
                    # 清掉已刪除項目的 checkbox state,避免下次出現殘留勾選
                    for fp in moved_paths:
                        st.session_state.pop(f"del_{fp}", None)
                    st.session_state.select_all = False

                    st.success(f"✅ 已將 {len(manifest)} 張移到 Trash,可以繼續挑選")
                    st.rerun()  # 全頁 rerun,讓卡片列表立即更新
            st.markdown("</div>", unsafe_allow_html=True)

            # ─── 分頁 + 卡片網格(在刪除按鈕下方)─────
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            paged_bad, _, _, _ = _paginate(results["Bad"], "bad", default_size=30)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            cols = st.columns(3, gap="small")
            for i, item in enumerate(paged_bad):
                with cols[i % 3]:
                    _render_bad_card(item)
        else:
            st.markdown("""
            <div class="empty">
                <div class="empty-glyph">✓</div>
                <div class="empty-title">No rejects.</div>
                <div class="empty-desc">
                    AI 沒有偵測到任何表情崩壞的照片。
                    你的相簿狀態完美 — 全部都通過。
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Good ──
    with tab_good:
        if results["Good"]:
            st.markdown(
                f"<div style='font-size:.92rem;color:#847f72;margin-bottom:10px;'>"
                f"依信心高到低排序</div>",
                unsafe_allow_html=True,
            )
            paged_good, _, _, _ = _paginate(results["Good"], "good", default_size=30)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            cols = st.columns(4, gap="small")
            for i, item in enumerate(paged_good):
                with cols[i % 4]:
                    thumb = load_thumbnail(item["path"], item["mtime"])
                    if thumb:
                        st.image(thumb, use_container_width=True)
                    st.markdown(
                        f"<span class='conf conf-good'>"
                        f"自然度 {item['prob']*100:.0f}%</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='filename' title='{html.escape(item['path'].name)}'>"
                        f"{html.escape(item['path'].name)}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown("""
            <div class="empty">
                <div class="empty-glyph">∅</div>
                <div class="empty-title">No selects.</div>
                <div class="empty-desc">
                    AI 沒有判定任何照片為 Good。
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── NoFace ──
    with tab_noface:
        if results["NoFace"]:
            st.markdown(
                f"<div style='font-size:.92rem;color:#847f72;margin-bottom:10px;'>"
                f"無法偵測到合格人臉的照片</div>",
                unsafe_allow_html=True,
            )
            paged_nf, _, _, _ = _paginate(results["NoFace"], "nf", default_size=30)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            cols = st.columns(4, gap="small")
            for i, item in enumerate(paged_nf):
                with cols[i % 4]:
                    thumb = load_thumbnail(item["path"], item["mtime"])
                    if thumb:
                        st.image(thumb, use_container_width=True)
                    st.markdown(
                        f"<div class='filename' title='{html.escape(item['path'].name)}'>"
                        f"{html.escape(item['path'].name)}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown("""
            <div class="empty">
                <div class="empty-glyph">◉</div>
                <div class="empty-title">All faces located.</div>
                <div class="empty-desc">
                    每張照片都成功偵測到人臉。
                </div>
            </div>
            """, unsafe_allow_html=True)
