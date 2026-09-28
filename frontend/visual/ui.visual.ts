import { test, expect, type Page } from '@playwright/test'

// Fixed server state covering every provider status the Settings show.
const CONFIG = {
  server_has_llm_key: true,
  scaleway_models: ['gemma-4-26b-a4b-it', 'deepseek-v4-flash-0731'],
  providers: {
    scaleway: { server_key: true, issue: null },
    gemini: { server_key: true, issue: null },
    mistral: { server_key: false, issue: null },
    openai: {
      server_key: false,
      issue: 'Document search uses Gemini embeddings, which need GOOGLE_API_KEY on the server.',
    },
    ollama: {
      server_key: false,
      issue: 'Local embeddings are not built yet. Run: LLM_PROVIDER=ollama uv run python ingestion.py',
    },
  },
}

const ANSWER = {
  answer:
    '**ALARA** means keeping exposure *as low as reasonably achievable*.\n\n- Time\n- Distance\n- Shielding',
  sources: [
    { source: 'GSR Part 3', document_type: 'IAEA' },
    { source: 'BEK nr 669 af 2019', document_type: 'Danish law' },
  ],
  chat_history: [['What is ALARA?', 'ALARA means keeping exposure as low as reasonably achievable.']],
  warning: null,
  used_web_search: false,
  used_web_search_label: null,
  privacy_mode: false,
}

async function openApp(page: Page) {
  await page.addInitScript(() => {
    localStorage.clear()
    localStorage.setItem('radiation-safety-model', 'scaleway')
  })
  await page.route('**/api/config', (r) => r.fulfill({ json: CONFIG }))
  await page.route('**/api/query', (r) => r.fulfill({ json: ANSWER }))
  await page.goto('/')
  await expect(page.getByRole('combobox')).toHaveValue('scaleway')
}

for (const scheme of ['light', 'dark'] as const) {
  test.describe(`${scheme} mode`, () => {
    test.use({ colorScheme: scheme, viewport: { width: 1100, height: 900 } })

    test('header controls', async ({ page }) => {
      await openApp(page)
      await expect(page.locator('.app-header')).toHaveScreenshot(`header-${scheme}.png`)
    })

    test('settings with every provider status', async ({ page }) => {
      await page.setViewportSize({ width: 1100, height: 2000 })
      await openApp(page)
      await page.getByRole('button', { name: 'Settings', exact: true }).click()
      const modal = page.locator('.settings-modal')
      await modal.evaluate((el) => {
        el.style.maxHeight = 'none'
      })
      await expect(modal).toHaveScreenshot(`settings-${scheme}.png`)
    })

    test('answer with sources', async ({ page }) => {
      await openApp(page)
      await page.getByPlaceholder(/Ask a question/i).fill('What is ALARA?')
      await page.getByRole('button', { name: 'Ask' }).click()
      await expect(page.getByText('Shielding')).toBeVisible()
      await expect(page).toHaveScreenshot(`answer-${scheme}.png`)
    })
  })
}
