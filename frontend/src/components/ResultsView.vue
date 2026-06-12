<script setup>
import { inject, ref, reactive, computed, watch } from 'vue'
import { api } from '../api.js'

const state = inject('state')
const showAlert = inject('showAlert')
const rememberTrashFolder = inject('rememberTrashFolder')

const activeTab = ref('bad')  // bad | good | noface
const selectedPaths = reactive(new Set())   // checkbox 勾選的 Bad 照片路徑
const pageBad = ref(1)
const pageGood = ref(1)
const pageNoFace = ref(1)
const pageSize = ref(30)
const isDeleting = ref(false)

const r = computed(() => state.scanResults || { Good: [], Bad: [], NoFace: [] })
const nBad = computed(() => r.value.Bad.length)
const nGood = computed(() => r.value.Good.length)
const nNoFace = computed(() => r.value.NoFace.length)
const nTotal = computed(() => nBad.value + nGood.value + nNoFace.value)

function paged(items, page) {
  const start = (page - 1) * pageSize.value
  return items.slice(start, start + pageSize.value)
}
function nPages(items) {
  return Math.max(1, Math.ceil(items.length / pageSize.value))
}

const pagedBad = computed(() => paged(r.value.Bad, pageBad.value))
const pagedGood = computed(() => paged(r.value.Good, pageGood.value))
const pagedNoFace = computed(() => paged(r.value.NoFace, pageNoFace.value))

// 切換 tab / 改頁時的計算
const badPages = computed(() => nPages(r.value.Bad))
const goodPages = computed(() => nPages(r.value.Good))
const nofacePages = computed(() => nPages(r.value.NoFace))

function toggleSelect(path) {
  if (selectedPaths.has(path)) selectedPaths.delete(path)
  else selectedPaths.add(path)
}
function selectAll() {
  r.value.Bad.forEach(it => selectedPaths.add(it.path))
}
function clearAll() {
  selectedPaths.clear()
}

async function doDelete() {
  if (selectedPaths.size === 0) {
    showAlert('warning', '還沒勾選任何照片喔!')
    return
  }
  isDeleting.value = true
  try {
    const result = await api.moveToTrash(
      state.scanFolder,
      Array.from(selectedPaths),
    )
    // 從 scanResults 移除已刪除的
    const movedSet = new Set(result.moved)
    state.scanResults.Bad = state.scanResults.Bad.filter(
      it => !movedSet.has(it.path)
    )
    selectedPaths.clear()
    rememberTrashFolder(state.scanFolder)
    showAlert('success', `✅ 已將 ${result.moved.length} 張移到 Trash`)
    // 修正頁碼避免空白頁
    if (pageBad.value > badPages.value) pageBad.value = badPages.value
  } catch (e) {
    showAlert('error', `刪除失敗: ${e.message}`)
  } finally {
    isDeleting.value = false
  }
}

// 截斷路徑顯示
function shortFolder(p) {
  if (!p) return ''
  return p.length > 50 ? '…' + p.slice(-47) : p
}
</script>

<template>
  <div>
    <div class="folder-bar">
      <span class="key">掃描路徑</span>
      <span class="path">{{ shortFolder(state.scanFolder) }}</span>
    </div>

    <!-- Stats overview -->
    <div class="stat-row">
      <div class="stat-card" style="--accent: var(--ink);">
        <div class="stat-label">總計掃描</div>
        <div class="stat-value">{{ nTotal }}</div>
      </div>
      <div class="stat-card" style="--accent: var(--crimson);">
        <div class="stat-label">建議刪除</div>
        <div class="stat-value">{{ nBad }}</div>
      </div>
      <div class="stat-card" style="--accent: var(--sage);">
        <div class="stat-label">完美表情</div>
        <div class="stat-value">{{ nGood }}</div>
      </div>
      <div class="stat-card" style="--accent: var(--ink-mute);">
        <div class="stat-label">未偵測到</div>
        <div class="stat-value">{{ nNoFace }}</div>
      </div>
    </div>

    <!-- Tabs -->
    <div class="tabs">
      <button class="tab" :class="{ active: activeTab === 'bad' }"
              @click="activeTab = 'bad'">⚠ 建議刪除 · {{ nBad }}</button>
      <button class="tab" :class="{ active: activeTab === 'good' }"
              @click="activeTab = 'good'">✓ 完美表情 · {{ nGood }}</button>
      <button class="tab" :class="{ active: activeTab === 'noface' }"
              @click="activeTab = 'noface'">○ 未偵測到 · {{ nNoFace }}</button>
    </div>

    <!-- BAD TAB -->
    <div v-if="activeTab === 'bad'">
      <template v-if="r.Bad.length">
        <!-- Action bar 在最上方 -->
        <div class="action-bar">
          <div class="count-display">
            <span class="num">{{ selectedPaths.size }}</span>
            <span class="lbl">張已準備刪除</span>
          </div>
          <div style="display: flex; gap: 8px;">
            <button class="btn btn-ghost" @click="selectAll" :disabled="isDeleting">全選</button>
            <button class="btn btn-ghost" @click="clearAll" :disabled="isDeleting">全清</button>
            <button class="btn btn-danger" @click="doDelete"
                    :disabled="isDeleting || selectedPaths.size === 0">
              🗑 移到 Trash
            </button>
          </div>
        </div>

        <!-- Pagination -->
        <div class="pagination">
          <select v-model="pageSize">
            <option :value="15">每頁 15</option>
            <option :value="30">每頁 30</option>
            <option :value="60">每頁 60</option>
            <option :value="120">每頁 120</option>
          </select>
          <button class="btn btn-ghost" :disabled="pageBad <= 1"
                  @click="pageBad = Math.max(1, pageBad - 1)">← 上一頁</button>
          <button class="btn btn-ghost" :disabled="pageBad >= badPages"
                  @click="pageBad = Math.min(badPages, pageBad + 1)">下一頁 →</button>
          <span class="info">第 <b>{{ pageBad }}</b> / {{ badPages }} 頁  ·  共 <b>{{ nBad }}</b> 張</span>
        </div>

        <!-- Cards grid -->
        <div class="photo-grid cols-3">
          <div v-for="item in pagedBad" :key="item.path" class="photo-card">
            <span class="conf conf-bad">崩壞機率 {{ (item.prob * 100).toFixed(1) }}%</span>
            <label class="check-row">
              <input
                type="checkbox"
                :checked="selectedPaths.has(item.path)"
                @change="toggleSelect(item.path)"
              />
              勾選刪除
            </label>
            <div class="img-pair">
              <div class="img-block">
                <img :src="api.imageUrl(item.path, 200)" :alt="item.name" loading="lazy" />
                <div class="cap">原圖</div>
              </div>
              <div class="img-block">
                <img :src="api.faceUrl(item.face_id)" :alt="item.name" loading="lazy" />
                <div class="cap">AI 鎖定區</div>
              </div>
            </div>
            <div class="filename" :title="item.name">{{ item.name }}</div>
          </div>
        </div>
      </template>
      <div v-else class="empty">
        <div class="empty-glyph">✓</div>
        <div class="empty-title">沒有發現廢片</div>
        <div class="empty-desc">AI 沒有偵測到任何表情崩壞的照片。</div>
      </div>
    </div>

    <!-- GOOD TAB -->
    <div v-if="activeTab === 'good'">
      <template v-if="r.Good.length">
        <div class="pagination">
          <select v-model="pageSize">
            <option :value="15">每頁 15</option>
            <option :value="30">每頁 30</option>
            <option :value="60">每頁 60</option>
            <option :value="120">每頁 120</option>
          </select>
          <button class="btn btn-ghost" :disabled="pageGood <= 1"
                  @click="pageGood = Math.max(1, pageGood - 1)">← 上一頁</button>
          <button class="btn btn-ghost" :disabled="pageGood >= goodPages"
                  @click="pageGood = Math.min(goodPages, pageGood + 1)">下一頁 →</button>
          <span class="info">第 <b>{{ pageGood }}</b> / {{ goodPages }} 頁  ·  共 <b>{{ nGood }}</b> 張</span>
        </div>
        <div class="photo-grid cols-4">
          <div v-for="item in pagedGood" :key="item.path" class="photo-card">
            <div class="img-solo">
              <img :src="api.imageUrl(item.path, 240)" :alt="item.name" loading="lazy" />
            </div>
            <span class="conf conf-good">✓ {{ (item.prob * 100).toFixed(0) }}% Good</span>
            <div class="filename" :title="item.name">{{ item.name }}</div>
          </div>
        </div>
      </template>
      <div v-else class="empty">
        <div class="empty-glyph">∅</div>
        <div class="empty-title">沒有完美表情</div>
        <div class="empty-desc">AI 沒有判定任何照片為 Good。</div>
      </div>
    </div>

    <!-- NO FACE TAB -->
    <div v-if="activeTab === 'noface'">
      <template v-if="r.NoFace.length">
        <div class="pagination">
          <select v-model="pageSize">
            <option :value="15">每頁 15</option>
            <option :value="30">每頁 30</option>
            <option :value="60">每頁 60</option>
          </select>
          <button class="btn btn-ghost" :disabled="pageNoFace <= 1"
                  @click="pageNoFace = Math.max(1, pageNoFace - 1)">← 上一頁</button>
          <button class="btn btn-ghost" :disabled="pageNoFace >= nofacePages"
                  @click="pageNoFace = Math.min(nofacePages, pageNoFace + 1)">下一頁 →</button>
          <span class="info">第 <b>{{ pageNoFace }}</b> / {{ nofacePages }} 頁  ·  共 <b>{{ nNoFace }}</b> 張</span>
        </div>
        <div class="photo-grid cols-4">
          <div v-for="item in pagedNoFace" :key="item.path" class="photo-card">
            <div class="img-solo">
              <img :src="api.imageUrl(item.path, 240)" :alt="item.name" loading="lazy" />
            </div>
            <div class="filename" :title="item.name">{{ item.name }}</div>
          </div>
        </div>
      </template>
      <div v-else class="empty">
        <div class="empty-glyph">◉</div>
        <div class="empty-title">全部都偵測到人臉</div>
        <div class="empty-desc">每張照片都成功定位到人臉。</div>
      </div>
    </div>
  </div>
</template>
