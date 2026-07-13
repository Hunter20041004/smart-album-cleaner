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

async function installOfflineBoundary(page, externalRequests) {
  const loopbackHosts = new Set(['127.0.0.1', 'localhost', '[::1]', '::1'])
  await page.route('**/*', async route => {
    const request = route.request()
    const url = new URL(request.url())
    if (
      (url.protocol === 'http:' || url.protocol === 'https:') &&
      !loopbackHosts.has(url.hostname)
    ) {
      externalRequests.push(`${request.method()} ${url.origin}${url.pathname}`)
      await route.abort('blockedbyclient')
      return
    }
    await route.fallback()
  })
}

async function installApiFixtureRouter(page, unexpectedRequests) {
  await page.route('**/api/**', async route => {
    const request = route.request()
    const pathname = new URL(request.url()).pathname
    const key = `${request.method()} ${pathname}`

    if (key === 'GET /api/health') {
      await route.fulfill({
        json: { status: 'ok', model_loaded: true, model_path: 'synthetic-model' },
      })
      return
    }
    if (key === 'POST /api/select-folder') {
      await route.fulfill({ json: { cancelled: false, folder: album } })
      return
    }
    if (key === 'POST /api/scan') {
      await route.fulfill({ json: { job_id: 'demo-job' } })
      return
    }
    if (key === 'GET /api/scan/demo-job') {
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
      return
    }
    if (key === 'GET /api/image') {
      await route.fulfill({
        contentType: 'image/svg+xml',
        body: solidSvg('#d7c7a5'),
      })
      return
    }
    if (request.method() === 'GET' && /^\/api\/face\/[^/]+$/.test(pathname)) {
      await route.fulfill({
        contentType: 'image/svg+xml',
        body: solidSvg('#b88a66'),
      })
      return
    }

    unexpectedRequests.push(key)
    await route.abort('blockedbyclient')
  })
}

test('blocks API requests outside the synthetic fixture allowlist', async ({ page }) => {
  const fallbackRequests = []
  const externalRequests = []
  const unexpectedRequests = []
  await page.route('**/api/**', route => {
    fallbackRequests.push(route.request().url())
    return route.fulfill({ status: 418, body: 'unexpected fallback' })
  })
  await installOfflineBoundary(page, externalRequests)
  await installApiFixtureRouter(page, unexpectedRequests)
  await page.goto('/')

  const result = await page.evaluate(async () => {
    try {
      const response = await fetch('/api/unexpected')
      return { blocked: false, status: response.status }
    } catch {
      return { blocked: true }
    }
  })

  expect(result).toEqual({ blocked: true })
  expect(fallbackRequests).toEqual([])
  expect(externalRequests).toEqual([])
  expect(unexpectedRequests).toEqual(['GET /api/unexpected'])
})

test('blocks non-loopback network requests', async ({ page }) => {
  const fallbackRequests = []
  const externalRequests = []
  const unexpectedRequests = []
  await page.route('https://external.invalid/**', route => {
    fallbackRequests.push(route.request().url())
    return route.fulfill({ status: 418, body: 'unexpected external fallback' })
  })
  await installOfflineBoundary(page, externalRequests)
  await installApiFixtureRouter(page, unexpectedRequests)
  await page.goto('/')

  const result = await page.evaluate(async () => {
    try {
      const response = await fetch('https://external.invalid/asset')
      return { blocked: false, status: response.status }
    } catch {
      return { blocked: true }
    }
  })

  expect(result).toEqual({ blocked: true })
  expect(fallbackRequests).toEqual([])
  expect(externalRequests).toEqual(['GET https://external.invalid/asset'])
  expect(unexpectedRequests).toEqual([])
})

test('captures the real review-before-trash flow with synthetic fixtures', async ({ page }) => {
  const pageErrors = []
  const externalRequests = []
  const unexpectedRequests = []
  page.on('pageerror', error => pageErrors.push(error.message))
  await installOfflineBoundary(page, externalRequests)
  await installApiFixtureRouter(page, unexpectedRequests)

  await page.goto('/')
  await expect(page.getByText('使用流程')).toBeVisible()

  await page.getByRole('button', { name: '選擇資料夾' }).click()
  await expect(page.getByTitle(album)).toBeVisible()

  await page.getByRole('button', { name: '啟動 AI 掃描' }).click()
  await expect(page.getByText('神經網路正在分析…')).toBeVisible()

  const badTab = page.getByRole('button', { name: '⚠ 建議檢視 · 3' })
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

  await page.screenshot({
    path: screenshotPath,
    animations: 'disabled',
    fullPage: false,
  })
  expect(pageErrors).toEqual([])
  expect(externalRequests).toEqual([])
  expect(unexpectedRequests).toEqual([])
})
