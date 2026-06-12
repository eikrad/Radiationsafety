import { test, expect } from '@playwright/test'

const MOCK_CONFIG = { server_has_llm_key: false }
const MOCK_QUERY_RESPONSE = {
  answer: 'Test answer about radiation safety.',
  sources: [],
  chat_history: [['What is ALARA?', 'Test answer about radiation safety.']],
  privacy_mode: true,
  used_web_search: false,
  warning: null,
  used_web_search_label: null,
}

test.beforeEach(async ({ page }) => {
  await page.route('/api/config', (r) => r.fulfill({ json: MOCK_CONFIG }))
  await page.route('/api/query', (r) => r.fulfill({ json: MOCK_QUERY_RESPONSE }))
  await page.goto('/')
  // Start with privacy mode off
  await page.evaluate(() => localStorage.removeItem('radiationsafety_enforce_privacy_mode'))
  await page.reload()
})

async function openSettings(page) {
  await page.locator('button[aria-label="Settings"]').click()
}

async function enablePrivacyMode(page) {
  await openSettings(page)
  const checkbox = page.getByRole('checkbox', { name: /privacy/i })
  await checkbox.check()
  await page.getByRole('button', { name: 'Save' }).click()
}

// --- Mocked UI Tests (always run) ---

test('privacy-toggle-shows-badge: enabling privacy mode shows 🔒 badge', async ({ page }) => {
  await expect(page.locator('.privacy-badge')).not.toBeVisible()
  await enablePrivacyMode(page)
  await expect(page.locator('.privacy-badge')).toBeVisible()
})

test('privacy-disables-cloud-models: non-ollama options disabled when privacy on', async ({ page }) => {
  await enablePrivacyMode(page)
  const select = page.getByRole('combobox')
  for (const model of ['mistral', 'gemini', 'openai']) {
    const option = select.locator(`option[value="${model}"]`)
    await expect(option).toBeDisabled()
  }
})

test('privacy-only-ollama-selectable: ollama option always enabled', async ({ page }) => {
  await enablePrivacyMode(page)
  const ollamaOption = page.getByRole('combobox').locator('option[value="ollama"]')
  await expect(ollamaOption).not.toBeDisabled()
})

test('privacy-toggle-off-enables-all: disabling privacy re-enables all models', async ({ page }) => {
  await enablePrivacyMode(page)
  // Now disable
  await openSettings(page)
  await page.getByRole('checkbox', { name: /privacy/i }).uncheck()
  await page.getByRole('button', { name: 'Save' }).click()

  const select = page.getByRole('combobox')
  await expect(page.locator('.privacy-badge')).not.toBeVisible()
  for (const model of ['mistral', 'gemini', 'openai']) {
    const option = select.locator(`option[value="${model}"]`)
    await expect(option).not.toBeDisabled()
  }
})

test('privacy-persists-after-reload: privacy mode survives page reload', async ({ page }) => {
  await enablePrivacyMode(page)
  await page.reload()
  await expect(page.locator('.privacy-badge')).toBeVisible()
  const mistralOption = page.getByRole('combobox').locator('option[value="mistral"]')
  await expect(mistralOption).toBeDisabled()
})

test('privacy-auto-switches-to-ollama: enabling privacy mode selects ollama automatically', async ({ page }) => {
  // Start with mistral selected (default)
  await page.getByRole('combobox').selectOption('mistral')
  await expect(page.getByRole('combobox')).toHaveValue('mistral')

  await enablePrivacyMode(page)

  // Model should have switched to ollama automatically
  await expect(page.getByRole('combobox')).toHaveValue('ollama')
})

// --- Integration Tests (require PLAYWRIGHT_INTEGRATION=true + ollama serve) ---

test('ollama-query-returns-answer: privacy mode query gets response', async ({ page }) => {
  test.skip(!process.env.PLAYWRIGHT_INTEGRATION, 'requires local backend + Ollama')
  // Remove mock to hit real backend
  await page.unrouteAll()
  await page.goto('/')
  await enablePrivacyMode(page)
  await page.getByPlaceholder(/Ask a question/i).fill('What is ALARA?')
  await page.getByRole('button', { name: 'Ask' }).click()
  await expect(page.locator('.response-content, .assistant-message')).toBeVisible({ timeout: 60000 })
})

test('ollama-no-web-search: privacy mode never triggers web search', async ({ page }) => {
  test.skip(!process.env.PLAYWRIGHT_INTEGRATION, 'requires local backend + Ollama')
  // Intercept query response and verify used_web_search=false
  let responseBody: Record<string, unknown> | null = null
  await page.route('/api/query', async (route) => {
    const response = await route.fetch()
    responseBody = await response.json()
    await route.fulfill({ response })
  })
  await page.goto('/')
  await enablePrivacyMode(page)
  await page.getByPlaceholder(/Ask a question/i).fill('What is ALARA?')
  await page.getByRole('button', { name: 'Ask' }).click()
  await page.waitForResponse('/api/query')
  expect(responseBody?.used_web_search).toBe(false)
  expect(responseBody?.privacy_mode).toBe(true)
})
