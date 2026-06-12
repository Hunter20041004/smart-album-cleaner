<script setup>
import { inject, ref, computed } from 'vue'
import { api } from '../api.js'

const state = inject('state')
const startScan = inject('startScan')
const showAlert = inject('showAlert')

const isScanning = computed(() => state.viewMode === 'scanning')
const sourceLabel = computed(() => {
  if (state.sourceMode === 'files') {
    return state.selectedPaths.length
      ? `已選擇 ${state.selectedPaths.length} 張圖片`
      : '尚未選擇圖片'
  }
  return state.folder || '尚未選擇資料夾'
})

async function chooseFolder() {
  if (isScanning.value) return
  try {
    const data = await api.selectFolder()
    if (data.cancelled) return
    state.folder = data.folder
    state.selectedPaths = []
    state.sourceMode = 'folder'
    state.scanFolder = null
  } catch (e) {
    showAlert('error', e.message)
  }
}

async function chooseFiles() {
  if (isScanning.value) return
  try {
    const data = await api.selectFiles()
    if (data.cancelled) return
    const mergedPaths = new Set([...state.selectedPaths, ...data.paths])
    state.selectedPaths = Array.from(mergedPaths)
    state.folder = ''
    state.sourceMode = 'files'
    state.scanFolder = null
  } catch (e) {
    showAlert('error', e.message)
  }
}

function clearSelectedFiles() {
  state.selectedPaths = []
  state.sourceMode = 'files'
  state.scanFolder = null
}

async function viewTrash() {
  if (!state.scanFolder && !state.folder && state.trashFolders.length === 0) {
    showAlert('warning', '尚未有可查看的 Trash 位置')
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

    <div class="nav-label">掃描來源</div>
    <div class="folder-picker">
      <div class="selected-folder" :title="sourceLabel">
        {{ sourceLabel }}
      </div>
      <div class="source-actions">
        <button
          class="btn btn-ghost"
          :disabled="isScanning"
          @click="chooseFolder"
        >
          選擇資料夾
        </button>
        <button
          class="btn btn-ghost"
          :disabled="isScanning"
          @click="chooseFiles"
        >
          選擇圖片
        </button>
      </div>
      <button
        v-if="state.sourceMode === 'files'"
        class="btn btn-ghost btn-block"
        :disabled="isScanning || state.selectedPaths.length === 0"
        @click="clearSelectedFiles"
      >
        清空所選照片
      </button>
    </div>

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
      :disabled="!state.scanFolder && !state.folder && state.trashFolders.length === 0"
      @click="viewTrash"
    >
      ♻ 查看 Trash
    </button>

    <div class="app-footer" style="margin-top: 24px; padding-top: 14px;">
      100% 本機運行 <span class="accent">·</span> MIT License
    </div>
  </aside>
</template>
