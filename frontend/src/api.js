// API client — 統一管理所有後端呼叫
const BASE = ''  // 同源(Vite 用 proxy,production FastAPI 直接服務)

async function _fetch(url, opts = {}) {
  const res = await fetch(BASE + url, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }
  return res.json()
}

export const api = {
  // 健康檢查 + 模型狀態
  health: () => _fetch('/api/health'),
  selectFolder: () => _fetch('/api/select-folder', { method: 'POST' }),
  selectFiles: () => _fetch('/api/select-files', { method: 'POST' }),

  // 掃描
  startScan: ({ folder = null, paths = null }) =>
    _fetch('/api/scan', {
      method: 'POST',
      body: JSON.stringify({ folder, paths }),
    }),
  getScan: (jobId) => _fetch(`/api/scan/${jobId}`),
  cancelScan: (jobId) => _fetch(`/api/scan/${jobId}`, { method: 'DELETE' }),

  // 影像服務(回傳 URL 給 <img>)
  imageUrl: (path, w = 200) =>
    `${BASE}/api/image?path=${encodeURIComponent(path)}&w=${w}`,
  faceUrl: (faceId) => `${BASE}/api/face/${faceId}`,

  // Trash
  trashList: (folder) =>
    _fetch(`/api/trash?folder=${encodeURIComponent(folder)}`),
  moveToTrash: (folder, paths) =>
    _fetch('/api/trash', {
      method: 'POST',
      body: JSON.stringify({ folder, paths }),
    }),
  restoreFromTrash: (folder, trashPaths = null) =>
    _fetch('/api/trash/restore', {
      method: 'POST',
      body: JSON.stringify({ folder, trash_paths: trashPaths }),
    }),
  moveTrashToSystemTrash: (folder, trashPaths = null) =>
    _fetch('/api/trash/system-delete', {
      method: 'POST',
      body: JSON.stringify({ folder, trash_paths: trashPaths }),
    }),
}
