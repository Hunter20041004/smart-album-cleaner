import { expect, test } from '@playwright/test'
import { fileURLToPath } from 'node:url'

const screenshotPath = fileURLToPath(
  new URL('../../docs/screenshots/dashboard.png', import.meta.url),
)

const album = 'Synthetic Album'
const items = {
  Bad: [
    { path: 'sample-001.jpg', name: 'sample-001.jpg', prob: 0.962, face_id: 'sample-001' },
    { path: 'sample-002.jpg', name: 'sample-002.jpg', prob: 0.914, face_id: 'sample-002' },
    { path: 'sample-003.jpg', name: 'sample-003.jpg', prob: 0.873, face_id: 'sample-003' },
  ],
  Good: [
    { path: 'sample-004.jpg', name: 'sample-004.jpg', prob: 0.948, face_id: 'sample-004' },
    { path: 'sample-005.jpg', name: 'sample-005.jpg', prob: 0.901, face_id: 'sample-005' },
  ],
  NoFace: [
    { path: 'sample-006.jpg', name: 'sample-006.jpg', prob: 0 },
  ],
}

function solidSvg(color) {
  return `<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240" viewBox="0 0 320 240"><rect width="320" height="240" fill="${color}"/></svg>`
}

test('captures the real review-before-trash flow with synthetic fixtures', async ({ page }) => {
  const pageErrors = []
  page.on('pageerror', error => pageErrors.push(error.message))

  await page.route('**/api/health', route => route.fulfill({
    json: { status: 'ok', model_loaded: true, model_path: 'synthetic-model' },
  }))
  await page.route('**/api/select-folder', route => route.fulfill({
    json: { cancelled: false, folder: album },
  }))
  await page.route('**/api/scan', route => route.fulfill({
    json: { job_id: 'demo-job' },
  }))
  await page.route('**/api/scan/demo-job', async route => {
    await new Promise(resolve => setTimeout(resolve, 150))
    await route.fulfill({
      json: {
        status: 'done',
        current: 6,
        total: 6,
        current_name: 'sample-006.jpg',
        eta_seconds: 0,
        folder: album,
        results: items,
      },
    })
  })
  await page.route('**/api/image?*', route => route.fulfill({
    contentType: 'image/svg+xml',
    body: solidSvg('#d7c7a5'),
  }))
  await page.route('**/api/face/*', route => route.fulfill({
    contentType: 'image/svg+xml',
    body: solidSvg('#b88a66'),
  }))

  await page.goto('/')
  await expect(page.getByText('使用流程')).toBeVisible()

  await page.getByRole('button', { name: '選擇資料夾' }).click()
  await expect(page.getByTitle(album)).toBeVisible()

  await page.getByRole('button', { name: '啟動 AI 掃描' }).click()
  await expect(page.getByText('神經網路正在分析…')).toBeVisible()

  const badTab = page.getByRole('button', { name: '⚠ 建議刪除 · 3' })
  const trashAction = page.getByRole('button', { name: '🗑 移到 Trash' })
  await expect(badTab).toBeVisible()
  await badTab.click()
  await page.getByRole('checkbox').first().check()
  await expect(trashAction).toBeVisible()
  await expect(trashAction).toBeEnabled()
  await expect(page.getByText('sample-001.jpg', { exact: true })).toBeVisible()
  await page.waitForFunction(() => (
    [...document.images].every(image => image.complete && image.naturalWidth > 0)
  ))

  expect(pageErrors).toEqual([])
  await page.screenshot({
    path: screenshotPath,
    animations: 'disabled',
    fullPage: false,
  })
})
