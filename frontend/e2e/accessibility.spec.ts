import AxeBuilder from '@axe-core/playwright'
import { test, expect, type Page } from '@playwright/test'

// WCAG 2.1 A/AA checks with axe-core, above all colour contrast, in every state a
// user sees and in both colour schemes. Unlike screenshots, these need no
// baselines: an intended design change passes as long as it stays readable.
// (Native <select> popups are drawn by the browser, outside the page, so axe
// cannot see them.)

const CONFIG = {
  server_has_llm_key: true,
  scaleway_models: ['gemma-4-26b-a4b-it', 'deepseek-v4-flash-0731'],
  providers: {
    scaleway: { server_key: true, issue: null },
    gemini: { server_key: true, issue: null },
    mistral: { server_key: false, issue: null },
    openai: { server_key: false, issue: 'Document search uses Gemini embeddings, which need GOOGLE_API_KEY on the server.' },
    ollama: { server_key: false, issue: 'Local embeddings are not built yet.' },
  },
}

const ANSWER = {
  answer: '**ALARA** means keeping exposure *as low as reasonably achievable*.\n\n- Time\n- Distance\n- Shielding',
  sources: [
    { source: 'GSR Part 3', document_type: 'IAEA' },
    { source: 'https://www.retsinformation.dk/eli/lta/2019/669', document_type: 'Danish law' },
  ],
  chat_history: [['What is ALARA?', 'ALARA means …']],
  warning: 'Die Websuche konnte keine ausreichend guten Quellen liefern.',
  used_web_search: true,
  used_web_search_label: 'Sources incl. web search',
  privacy_mode: false,
}

async function openApp(page: Page, query?: (route: Parameters<Parameters<Page['route']>[1]>[0]) => void) {
  await page.addInitScript(() => localStorage.setItem('radiation-safety-model', 'scaleway'))
  await page.route('**/api/config', (r) => r.fulfill({ json: CONFIG }))
  if (query) await page.route('**/api/query', query)
  await page.goto('/')
  await expect(page.locator('[data-config="loaded"]')).toBeAttached()
}

const SOURCES = {
  sources: [
    {
      id: 'gsr-3',
      name: 'GSR Part 3',
      url: 'https://www.iaea.org/publications/8930',
      folder: 'IAEA',
      current_version: '2014',
      remote_version: '2014',
      update_available: false,
      local_date: null,
      remote_date: null,
      download_url: '',
      has_local_file: true,
      error: null,
    },
  ],
}

/** The Documents panel, opened after an ingestion that ended as ingestStatus says. */
async function openDocuments(page: Page, ingestStatus: object) {
  await page.route('**/api/documents/check-updates', (r) => r.fulfill({ json: SOURCES }))
  await page.route('**/api/ingest/status', (r) => r.fulfill({ json: ingestStatus }))
  await openApp(page)
  await page.getByRole('button', { name: 'Documents and updates' }).click()
  await expect(page.getByText('GSR Part 3')).toBeVisible()
}

async function ask(page: Page) {
  await page.getByPlaceholder(/Ask a question/i).fill('What is ALARA?')
  await page.getByRole('button', { name: 'Ask' }).click()
}

async function expectNoViolations(page: Page) {
  const { violations } = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze()
  const summary = violations.map((v) => ({
    rule: v.id,
    impact: v.impact,
    help: v.help,
    nodes: v.nodes.map((n) => `${n.target.join(' ')}: ${n.failureSummary?.split('\n').pop()?.trim()}`),
  }))
  expect(summary).toEqual([])
}

for (const scheme of ['light', 'dark'] as const) {
  test.describe(`${scheme} mode`, () => {
    test.use({ colorScheme: scheme })

    test('start page is readable', async ({ page }) => {
      await openApp(page)
      await expectNoViolations(page)
    })

    test('settings are readable', async ({ page }) => {
      await openApp(page)
      await page.getByRole('button', { name: 'Settings', exact: true }).click()
      await expect(page.getByRole('region', { name: 'Providers' })).toBeVisible()
      await expectNoViolations(page)
    })

    test('an answer with warning and sources is readable', async ({ page }) => {
      await openApp(page, (r) => r.fulfill({ json: ANSWER }))
      await ask(page)
      await expect(page.getByText('Shielding')).toBeVisible()
      await expectNoViolations(page)
    })

    test('the progress while waiting is readable', async ({ page }) => {
      await openApp(page, () => {}) // never answers
      await ask(page)
      await expect(page.getByRole('status')).toBeVisible()
      await expectNoViolations(page)
    })

    test('an error is readable', async ({ page }) => {
      await openApp(page, (r) =>
        r.fulfill({ status: 504, json: { detail: 'Scaleway did not answer within 60 s.' } })
      )
      await ask(page)
      await expect(page.getByText(/did not answer/)).toBeVisible()
      await expectNoViolations(page)
    })

    test('a failed ingestion is readable', async ({ page }) => {
      await openDocuments(page, {
        status: 'idle',
        last_error:
          'DoclingFailedError: docling could not read vejledning.pdf, also not on the CPU: RuntimeError: CUDA out of memory.',
        pdf_problems: [],
      })
      await expect(page.getByRole('alert')).toBeVisible()
      await expectNoViolations(page)
    })

    test('the PDFs an ingestion did not read as usual are readable', async ({ page }) => {
      await openDocuments(page, {
        status: 'idle',
        last_error: null,
        pdf_problems: [
          { file: 'vejledning.pdf', problem: 'read on the CPU' },
          { file: 'scannet.pdf', problem: 'almost no text (40 characters on 41 pages)' },
        ],
      })
      await expect(page.getByRole('list', { name: 'PDFs not read as usual' })).toBeVisible()
      await expectNoViolations(page)
    })

    test('a phone-sized screen is readable', async ({ page }) => {
      await page.setViewportSize({ width: 390, height: 844 })
      await openApp(page, (r) => r.fulfill({ json: ANSWER }))
      await ask(page)
      await expect(page.getByText('Shielding')).toBeVisible()
      await expectNoViolations(page)
    })
  })
}
