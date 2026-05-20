<script setup>
const steps = [
  { num: 1, title: '輸入路徑',  desc: '在左側填入相簿資料夾。' },
  { num: 2, title: '啟動掃描',  desc: '點擊「啟動 AI 掃描」,神經網路逐張分析。' },
  { num: 3, title: '挑出廢片',  desc: '在 Bad 分頁勾選你同意刪除的照片。' },
  { num: 4, title: '安全刪除',  desc: '軟刪除到 Trash,可隨時還原,不會遺失。' },
]

const classes = [
  { glyph: '✓', tag: 'GOOD',   name: '完美表情',     desc: '表情自然、雙眼張開、清晰可見',
    accent: 'var(--sage)',     tint: 'var(--sage-tint)' },
  { glyph: '⚠', tag: 'BAD',    name: '建議刪除',     desc: '閉眼、嘴歪、表情扭曲、明顯模糊',
    accent: 'var(--crimson)',  tint: 'var(--crimson-tint)' },
  { glyph: '○', tag: 'NOFACE', name: '未偵測到人臉', desc: '背影、風景、寵物、或臉太小',
    accent: 'var(--ink-mute)', tint: 'rgba(132,126,111,0.12)' },
]
</script>

<template>
  <div class="hero">
    <span class="hero-eyebrow">AI 相簿清理工具 · v2.0</span>
    <div class="hero-title">
      自動挑出相簿裡的<span class="accent">表情崩壞照片</span>,<br />
      一鍵軟刪除。
    </div>
    <div class="hero-lead">
      用自訓的 <b>MobileNetV3-Large</b> 神經網路,從你的相簿挑出閉眼、嘴歪、模糊等廢片。
      <b>本機運行,完全離線</b> — 你的照片不會離開這台電腦。
    </div>
  </div>

  <div class="section-head">
    <span class="num">01</span>
    <span class="title">使用流程</span>
    <span class="sub-en">Workflow</span>
  </div>
  <div class="workflow">
    <div v-for="s in steps" :key="s.num" class="workflow-step">
      <div class="num">{{ s.num }}</div>
      <div class="title">{{ s.title }}</div>
      <div class="desc">{{ s.desc }}</div>
    </div>
  </div>

  <div class="section-head">
    <span class="num">02</span>
    <span class="title">AI 怎麼分類</span>
    <span class="sub-en">Classification</span>
  </div>
  <div class="classifier-grid">
    <div v-for="c in classes" :key="c.tag"
         class="class-card"
         :style="{ '--accent': c.accent, '--accent-tint': c.tint }">
      <div class="icon-wrap">{{ c.glyph }}</div>
      <div class="tag">{{ c.tag }}</div>
      <div class="name">{{ c.name }}</div>
      <div class="desc">{{ c.desc }}</div>
    </div>
  </div>

  <div class="app-footer">
    Darkroom <span class="accent">·</span> Built with FastAPI + Vue 3
    <span class="accent">·</span> MIT License
  </div>
</template>
