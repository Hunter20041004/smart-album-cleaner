<script setup>
import { ref, reactive, provide, onMounted } from 'vue'
import Sidebar from './components/Sidebar.vue'
import WelcomePage from './components/WelcomePage.vue'
import ScanProgress from './components/ScanProgress.vue'
import ResultsView from './components/ResultsView.vue'
import TrashView from './components/TrashView.vue'
import { api } from './api.js'

// 全局狀態(用 reactive 管理)
const state = reactive({
  modelLoaded: false,
  folder: '',
  selectedPaths: [],
  sourceMode: 'folder',  // folder | files
  trashFolders: [],
  viewMode: 'welcome',   // welcome | scanning | results | trash
  jobId: null,
  scanProgress: null,    // { current, total, current_name, eta_seconds, status }
  scanResults: null,     // { Good, Bad, NoFace, errors }
  scanFolder: null,      // 已掃描的資料夾
  alert: null,           // { type: success|error|warning, message }
})

provide('state', state)

function rememberTrashFolder(folder) {
  if (!folder) return
  const folders = [folder, ...state.trashFolders.filter(f => f !== folder)]
  state.trashFolders = folders.slice(0, 20)
  localStorage.setItem('darkroom.trashFolders', JSON.stringify(state.trashFolders))
}
provide('rememberTrashFolder', rememberTrashFolder)

// 顯示提示訊息(3 秒後自動消失)
function showAlert(type, message) {
  state.alert = { type, message }
  setTimeout(() => {
    if (state.alert?.message === message) state.alert = null
  }, 4000)
}
provide('showAlert', showAlert)

// 健康檢查 — 確認後端在跑、模型有載入
onMounted(async () => {
  try {
    state.trashFolders = JSON.parse(localStorage.getItem('darkroom.trashFolders') || '[]')
  } catch {
    state.trashFolders = []
  }

  try {
    const h = await api.health()
    state.modelLoaded = h.model_loaded
    if (!h.model_loaded) {
      showAlert('error', '模型檔不存在,請先放到 models/mobilenet_face.pth')
    }
  } catch (e) {
    showAlert('error', '無法連接後端 API(uvicorn 在跑嗎?)')
  }
})

// 啟動掃描
async function startScan() {
  if (!state.folder && state.selectedPaths.length === 0) {
    showAlert('warning', '請先選擇資料夾或圖片')
    return
  }
  try {
    const payload = state.sourceMode === 'files'
      ? { paths: state.selectedPaths }
      : { folder: state.folder }
    const { job_id } = await api.startScan(payload)
    state.jobId = job_id
    state.viewMode = 'scanning'
    state.scanProgress = {
      current: 0, total: 0, current_name: '正在啟動...',
      eta_seconds: 0, status: 'running',
    }
    pollScanProgress()
  } catch (e) {
    showAlert('error', e.message)
  }
}
provide('startScan', startScan)

async function pollScanProgress() {
  if (!state.jobId) return
  try {
    const data = await api.getScan(state.jobId)
    state.scanProgress = data
    if (data.status === 'done') {
      state.scanResults = data.results
      state.scanFolder = data.folder
      rememberTrashFolder(data.folder)
      state.viewMode = 'results'
      state.jobId = null
      showAlert('success', `掃描完成 — Good ${data.results.Good.length} · Bad ${data.results.Bad.length} · NoFace ${data.results.NoFace.length}`)
    } else if (data.status === 'error') {
      showAlert('error', `掃描失敗: ${data.error}`)
      state.viewMode = 'welcome'
      state.jobId = null
    } else if (data.status === 'cancelled') {
      state.viewMode = 'welcome'
      state.jobId = null
    } else {
      // 還在跑,500ms 後再 poll
      setTimeout(pollScanProgress, 500)
    }
  } catch (e) {
    showAlert('error', e.message)
    state.viewMode = 'welcome'
    state.jobId = null
  }
}
</script>

<template>
  <Sidebar />
  <main class="main">
    <div v-if="state.alert" class="alert" :class="`alert-${state.alert.type}`">
      {{ state.alert.message }}
    </div>

    <WelcomePage v-if="state.viewMode === 'welcome'" />
    <ScanProgress v-else-if="state.viewMode === 'scanning'" />
    <ResultsView v-else-if="state.viewMode === 'results'" />
    <TrashView v-else-if="state.viewMode === 'trash'" />
  </main>
</template>
