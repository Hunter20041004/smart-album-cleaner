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
const isDeleting = ref(false)

const trashFolders = computed(() => {
  return Array.from(new Set([
    state.scanFolder,
    state.folder,
    ...state.trashFolders,
  ].filter(Boolean)))
})

const pagedItems = computed(() => {
  const start = (page.value - 1) * pageSize.value
  return items.value.slice(start, start + pageSize.value)
})
const nPages = computed(() =>
  Math.max(1, Math.ceil(items.value.length / pageSize.value))
)

async function load() {
  if (trashFolders.value.length === 0) return
  loading.value = true
  try {
    const results = await Promise.all(
      trashFolders.value.map(async (folder) => {
        const data = await api.trashList(folder)
        return data.items.map(item => ({
          ...item,
          folder,
          key: `${folder}::${item.trash_path}`,
        }))
      })
    )
    items.value = results.flat()
      .sort((a, b) => (b.deleted_at || '').localeCompare(a.deleted_at || ''))
    total.value = items.value.length
    selectedPaths.clear()
  } catch (e) {
    showAlert('error', `無法讀取 Trash: ${e.message}`)
  } finally {
    loading.value = false
  }
}

function selectedItems(all = false) {
  if (all) return items.value
  return items.value.filter(item => selectedPaths.has(item.key))
}

function groupedByFolder(targetItems) {
  const groups = new Map()
  for (const item of targetItems) {
    if (!groups.has(item.folder)) groups.set(item.folder, [])
    groups.get(item.folder).push(item.trash_path)
  }
  return groups
}

function toggleSelect(item) {
  if (selectedPaths.has(item.key)) selectedPaths.delete(item.key)
  else selectedPaths.add(item.key)
}
function selectAll() { items.value.forEach(it => selectedPaths.add(it.key)) }
function clearAll() { selectedPaths.clear() }

async function doRestore(restoreAll = false) {
  if (!restoreAll && selectedPaths.size === 0) {
    showAlert('warning', '還沒選任何照片')
    return
  }
  isRestoring.value = true
  try {
    let restored = 0
    for (const [folder, trashPaths] of groupedByFolder(selectedItems(restoreAll))) {
      const r = await api.restoreFromTrash(folder, trashPaths)
      restored += r.restored
    }
    showAlert('success', `✅ 已還原 ${restored} 張`)
    selectedPaths.clear()
    await load()  // 重新載入 Trash 列表
  } catch (e) {
    showAlert('error', `還原失敗: ${e.message}`)
  } finally {
    isRestoring.value = false
  }
}

async function doSystemDelete(deleteAll = false) {
  if (!deleteAll && selectedPaths.size === 0) {
    showAlert('warning', '還沒選任何照片')
    return
  }
  const count = deleteAll ? total.value : selectedPaths.size
  if (!confirm(`確定要把 ${count} 張照片移到電腦垃圾桶嗎？`)) return

  isDeleting.value = true
  try {
    let deleted = 0
    for (const [folder, trashPaths] of groupedByFolder(selectedItems(deleteAll))) {
      const r = await api.moveTrashToSystemTrash(folder, trashPaths)
      deleted += r.deleted
    }
    showAlert('success', `已移到電腦垃圾桶 ${deleted} 張`)
    selectedPaths.clear()
    await load()
  } catch (e) {
    showAlert('error', `移到電腦垃圾桶失敗: ${e.message}`)
  } finally {
    isDeleting.value = false
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
              :disabled="isRestoring || isDeleting || total === 0">
        ⤺ 還原全部 ({{ total }})
      </button>
      <button class="btn btn-danger" @click="doSystemDelete(true)"
              :disabled="isRestoring || isDeleting || total === 0">
        移到電腦垃圾桶 ({{ total }})
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
          <span class="lbl">張已勾選</span>
        </div>
        <div style="display: flex; gap: 8px;">
          <button class="btn btn-ghost" @click="selectAll" :disabled="isRestoring || isDeleting">全選</button>
          <button class="btn btn-ghost" @click="clearAll" :disabled="isRestoring || isDeleting">全清</button>
          <button class="btn btn-primary" @click="doRestore(false)"
                  :disabled="isRestoring || isDeleting || selectedPaths.size === 0">
            ⤺ 還原勾選的 {{ selectedPaths.size }} 張
          </button>
          <button class="btn btn-danger" @click="doSystemDelete(false)"
                  :disabled="isRestoring || isDeleting || selectedPaths.size === 0">
            移到電腦垃圾桶 {{ selectedPaths.size }} 張
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
              :checked="selectedPaths.has(item.key)"
              @change="toggleSelect(item)"
            />
            勾選
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
