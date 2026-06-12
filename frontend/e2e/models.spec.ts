import { test, expect } from '@playwright/test'

const MOCK_QUERY_RESPONSE = (privacyMode: boolean) => ({
  answer: 'Test answer about radiation safety.',
  sources: [],
  chat_history: [['What is ALARA?', 'Test answer about radiation safety.']],
  privacy_mode: privacyMode,
  used_web_search: false,
  warning: null,
  used_web_search_label: null,
})

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => localStorage.removeItem('radiationsafety_enforce_privacy_mode'))
  await page.route('/api/query', (r) => r.fulfill({ json: MOCK_QUERY_RESPONSE(false) }))
})

// --- Mocked UI Tests (always run) ---

test('all-four-models-in-dropdown: selector shows all providers', async ({ page }) => {
  await page.route('/api/config', (r) => r.fulfill({ json: { server_has_llm_key: true } }))
  await page.goto('/')
  const options = await page.getByRole('combobox').locator('option').allTextContents()
  expect(options).toContain('Mistral')
  expect(options).toContain('Gemini')
  expect(options).toContain('OpenAI')
  expect(options).toContain('Ollama (Local)')
})

test('api-key-hint-when-no-key: shows hint when no keys configured', async ({ page }) => {
  await page.route('/api/config', (r) => r.fulfill({ json: { server_has_llm_key: false } }))
  await page.goto('/')
  await expect(page.locator('.api-keys-hint')).toBeVisible()
})

test('api-key-hint-hidden-when-server-key: no hint when server has key', async ({ page }) => {
  await page.route('/api/config', (r) => r.fulfill({ json: { server_has_llm_key: true } }))
  await page.goto('/')
  await expect(page.locator('.api-keys-hint')).not.toBeVisible()
})

test('gemini-query-no-privacy-badge: Gemini queries show no privacy badge', async ({ page }) => {
  await page.route('/api/config', (r) => r.fulfill({ json: { server_has_llm_key: true } }))
  await page.goto('/')
  await page.getByRole('combobox').selectOption('gemini')
  await page.getByPlaceholder(/Ask a question/i).fill('What is ALARA?')
  await page.getByRole('button', { name: 'Ask' }).click()
  await page.waitForResponse('/api/query')
  await expect(page.locator('.privacy-badge')).not.toBeVisible()
})

test('ollama-query-shows-privacy-badge: Ollama queries show privacy badge', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('radiationsafety_enforce_privacy_mode', 'true'))
  await page.route('/api/config', (r) => r.fulfill({ json: { server_has_llm_key: true } }))
  await page.route('/api/query', (r) => r.fulfill({ json: MOCK_QUERY_RESPONSE(true) }))
  await page.goto('/')
  await page.getByPlaceholder(/Ask a question/i).fill('What is ALARA?')
  await page.getByRole('button', { name: 'Ask' }).click()
  await page.waitForResponse('/api/query')
  await expect(page.locator('.privacy-badge')).toBeVisible()
})

test('error-display-on-api-failure: shows error message when query fails', async ({ page }) => {
  await page.route('/api/config', (r) => r.fulfill({ json: { server_has_llm_key: true } }))
  await page.route('/api/query', (r) =>
    r.fulfill({ status: 400, json: { detail: 'Please provide a valid API key for Gemini in Settings.' } })
  )
  await page.goto('/')
  await page.getByPlaceholder(/Ask a question/i).fill('What is ALARA?')
  await page.getByRole('button', { name: 'Ask' }).click()
  await expect(page.locator('p.error')).toBeVisible()
  await expect(page.locator('p.error')).toContainText(/API key/i)
})

// --- Integration Tests (require PLAYWRIGHT_INTEGRATION=true + backend running) ---

for (const [provider, keyEnv, modelLabel] of [
  ['gemini', 'GOOGLE_API_KEY', 'Gemini'],
  ['mistral', 'MISTRAL_API_KEY', 'Mistral'],
] as const) {
  test(`${provider}-query-end-to-end: full query flow`, async ({ page }) => {
    test.skip(
      !process.env.PLAYWRIGHT_INTEGRATION || !process.env[keyEnv],
      `requires backend + ${keyEnv}`
    )
    await page.unrouteAll()
    await page.route('/api/config', (r) => r.fulfill({ json: { server_has_llm_key: false } }))
    await page.goto('/')
    await page.getByRole('combobox').selectOption(provider)
    // Enter API key via settings
    await page.locator('button[aria-label="Settings"]').click()
    await page.locator(`#api-key-${provider}`).fill(process.env[keyEnv]!)
    await page.getByRole('button', { name: 'Save' }).click()
    await page.getByPlaceholder(/Ask a question/i).fill('What is ALARA?')
    await page.getByRole('button', { name: 'Ask' }).click()
    await expect(page.locator('.response-content, .assistant-message')).toBeVisible({ timeout: 60000 })
  })
}

test('invalid-api-key-shows-error: wrong key returns error message', async ({ page }) => {
  test.skip(!process.env.PLAYWRIGHT_INTEGRATION, 'requires backend')
  await page.unrouteAll()
  await page.route('/api/config', (r) => r.fulfill({ json: { server_has_llm_key: false } }))
  await page.goto('/')
  await page.locator('button[aria-label="Settings"]').click()
  await page.locator('#api-key-gemini').fill('invalid-key-xyz')
  await page.getByRole('button', { name: 'Save' }).click()
  await page.getByRole('combobox').selectOption('gemini')
  await page.getByPlaceholder(/Ask a question/i).fill('What is ALARA?')
  await page.getByRole('button', { name: 'Ask' }).click()
  await expect(page.locator('p.error')).toBeVisible({ timeout: 15000 })
  await expect(page.locator('p.error')).toContainText(/API key/i)
})
