<script setup>
import { inject, ref, reactive, computed, onMounted } from 'vue'
import { api } from '../api.js'

const state = inject('state')
const showAlert = inject('showAlert')

const items = ref([])
const total = ref(0)
const selectedPaths = reactive(new Set())
const page = ref(1)
const pageSize = ref(20)
const loading = ref(false)
const isRestoring = ref(false)

// 用 scanFolder 優先,沒有就用 folder
const folder = computed(() => state.scanFolder || state.folder)

const pagedItems = computed(() => {
  const start = (page.value - 1) * pageSize.value
  return items.value.slice(start, start + pageSize.value)
})
const nPages = computed(() =>
  Math.max(1, Math.ceil(items.value.length / pageSize.value))
)

async function load() {
  if (!folder.value) return
  loading.value = true
  try {
    const data = await api.trashList(folder.value)
    items.value = data.items
    total.value = data.total
  } catch (e) {
    showAlert('error', `無法讀取 Trash: ${e.message}`)
  } finally {
    loading.value = false
  }
}

function toggleSelect(path) {
  if (selectedPaths.has(path)) selectedPaths.delete(path)
  else selectedPaths.add(path)
}
function selectAll() { items.value.forEach(it => selectedPaths.add(it.trash_path)) }
function clearAll() { selectedPaths.clear() }

async function doRestore(restoreAll = false) {
  if (!restoreAll && selectedPaths.size === 0) {
    showAlert('warning', '還沒選任何照片')
    return
  }
  isRestoring.value = true
  try {
    const trashPaths = restoreAll ? null : Array.from(selectedPaths)
    const r = await api.restoreFromTrash(folder.value, trashPaths)
    showAlert('success', `✅ 已還原 ${r.restored} 張`)
    selectedPaths.clear()
    await load()  // 重新載入 Trash 列表
  } catch (e) {
    showAlert('error', `還原失敗: ${e.message}`)
  } finally {
    isRestoring.value = false
  }
}

function back() {
  if (state.scanResults) state.viewMode = 'results'
  else state.viewMode = 'welcome'
}

onMounted(load)
</script>

<template>
  <div>
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 18px;">
      <button class="btn btn-ghost" @click="back">← 返回</button>
      <h2 style="margin: 0; flex: 1;">♻ Trash · {{ total }} 張照片</h2>
      <button class="btn btn-ghost" @click="doRestore(true)"
              :disabled="isRestoring || total === 0">
        ⤺ 還原全部 ({{ total }})
      </button>
    </div>

    <template v-if="loading">
      <div class="empty"><div class="empty-glyph">…</div><div class="empty-title">載入中</div></div>
    </template>

    <template v-else-if="items.length">
      <!-- Action bar -->
      <div class="action-bar">
        <div class="count-display">
          <span class="num" style="color: var(--gold);">{{ selectedPaths.size }}</span>
          <span class="lbl">張準備還原</span>
        </div>
        <div style="display: flex; gap: 8px;">
          <button class="btn btn-ghost" @click="selectAll" :disabled="isRestoring">全選</button>
          <button class="btn btn-ghost" @click="clearAll" :disabled="isRestoring">全清</button>
          <button class="btn btn-primary" @click="doRestore(false)"
                  :disabled="isRestoring || selectedPaths.size === 0">
            ⤺ 還原勾選的 {{ selectedPaths.size }} 張
          </button>
        </div>
      </div>

      <!-- Pagination -->
      <div class="pagination">
        <select v-model="pageSize">
          <option :value="20">每頁 20</option>
          <option :value="40">每頁 40</option>
          <option :value="80">每頁 80</option>
        </select>
        <button class="btn btn-ghost" :disabled="page <= 1" @click="page--">← 上一頁</button>
        <button class="btn btn-ghost" :disabled="page >= nPages" @click="page++">下一頁 →</button>
        <span class="info">第 <b>{{ page }}</b> / {{ nPages }} 頁  ·  共 <b>{{ total }}</b> 張</span>
      </div>

      <div class="photo-grid cols-4">
        <div v-for="item in pagedItems" :key="item.trash_path" class="photo-card">
          <label class="check-row">
            <input
              type="checkbox"
              :checked="selectedPaths.has(item.trash_path)"
              @change="toggleSelect(item.trash_path)"
            />
            勾選還原
          </label>
          <div class="img-solo" v-if="item.exists">
            <img :src="api.imageUrl(item.trash_path, 240)" :alt="item.name" loading="lazy" />
          </div>
          <div v-else class="alert alert-warning" style="margin: 0;">⚠ 檔案已不存在</div>
          <div class="filename" :title="item.name">{{ item.name }}</div>
          <div class="meta">刪除於 {{ item.deleted_at.slice(0, 16).replace('T', ' ') }}</div>
        </div>
      </div>
    </template>

    <div v-else class="empty">
      <div class="empty-glyph">◉</div>
      <div class="empty-title">Trash 是空的</div>
      <div class="empty-desc">還沒有照片被刪除,或已全部還原。</div>
    </div>
  </div>
</template>
