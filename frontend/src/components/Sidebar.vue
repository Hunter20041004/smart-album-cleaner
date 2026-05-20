<script setup>
import { inject, ref, computed } from 'vue'
import { api } from '../api.js'

const state = inject('state')
const startScan = inject('startScan')
const showAlert = inject('showAlert')

const isScanning = computed(() => state.viewMode === 'scanning')

async function viewTrash() {
  if (!state.scanFolder && !state.folder) {
    showAlert('warning', '尚未指定資料夾')
    return
  }
  state.viewMode = 'trash'
}

function resetScan() {
  state.scanResults = null
  state.viewMode = 'welcome'
}
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">Darkroom<span class="dot">.</span></div>
      <div class="brand-meta">AI 表情相簿管家</div>
    </div>

    <div class="nav-label">目標資料夾</div>
    <input
      type="text"
      v-model="state.folder"
      placeholder="例:./datasets/raw"
      :disabled="isScanning"
    />

    <div class="nav-label">啟動</div>
    <button
      class="btn btn-primary btn-block"
      :disabled="isScanning || !state.modelLoaded"
      @click="startScan"
    >
      🚀 啟動 AI 掃描
    </button>

    <template v-if="state.scanResults">
      <button class="btn btn-ghost btn-block" style="margin-top: 8px" @click="resetScan">
        🔄 清空結果
      </button>
    </template>

    <button
      class="btn btn-ghost btn-block"
      style="margin-top: 10px"
      :disabled="!state.scanFolder && !state.folder"
      @click="viewTrash"
    >
      ♻ 查看 Trash
    </button>

    <div class="app-footer" style="margin-top: 24px; padding-top: 14px;">
      100% 本機運行 <span class="accent">·</span> MIT License
    </div>
  </aside>
</template>
