<script setup>
import { inject, computed } from 'vue'

const state = inject('state')

const percent = computed(() => {
  const p = state.scanProgress
  if (!p || !p.total) return 0
  return Math.min(100, (p.current / p.total) * 100)
})

const etaStr = computed(() => {
  const sec = state.scanProgress?.eta_seconds || 0
  if (sec < 1) return '—'
  if (sec < 60) return `${Math.round(sec)}s`
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m}m ${String(s).padStart(2, '0')}s`
})
</script>

<template>
  <div class="scan-card" v-if="state.scanProgress">
    <div class="scan-label">
      <span class="pulse-dot"></span>神經網路正在分析…
    </div>
    <div>
      <div class="scan-filename">{{ state.scanProgress.current_name || '準備中...' }}</div>
    </div>
    <div class="scan-bar">
      <div class="scan-bar-fill" :style="{ width: percent + '%' }"></div>
    </div>
    <div class="scan-stats">
      <div>
        <div class="scan-stat-num">{{ state.scanProgress.current }}</div>
        <div class="scan-stat-lbl">共 {{ state.scanProgress.total }} 張</div>
      </div>
      <div>
        <div class="scan-stat-num accent">{{ percent.toFixed(1) }}%</div>
        <div class="scan-stat-lbl">完成度</div>
      </div>
      <div>
        <div class="scan-stat-num">{{ etaStr }}</div>
        <div class="scan-stat-lbl">預估剩餘</div>
      </div>
    </div>
  </div>
</template>
