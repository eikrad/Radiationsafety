import { act, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DocumentsPanel } from './DocumentsPanel'

type Body = Record<string, unknown>

/** The API with no document sources; GET /ingest/status answers with these
 * bodies in turn, the last one from then on. */
function stubApi(ingestStatuses: [Body, ...Body[]]) {
  const statuses: [Body, ...Body[]] = [...ingestStatuses]
  const json = (body: Body, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (url.endsWith('/ingest/status')) return json(statuses.length > 1 ? statuses.shift()! : statuses[0])
      if (url.endsWith('/ingest')) return json({ status: 'accepted' }, 202)
      if (url.endsWith('/documents/check-updates')) return json({ sources: [] })
      return json({ detail: 'Not Found' }, 404)
    }),
  )
}

/** Start an ingestion and let the panel poll its status once. */
async function startIngestion() {
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
  render(<DocumentsPanel onClose={() => {}} />)
  await user.click(screen.getByRole('button', { name: 'Start ingestion' }))
  await act(() => vi.advanceTimersByTimeAsync(3000))
}

describe('the outcome of an ingestion', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('says why an ingestion failed', async () => {
    stubApi([
      {
        status: 'idle',
        last_error: 'DoclingFailedError: docling could not read vejledning.pdf, also not on the CPU',
        pdf_problems: [],
      },
    ])

    await startIngestion()

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Ingestion failed: DoclingFailedError: docling could not read vejledning.pdf, also not on the CPU',
    )
    expect(screen.queryByText('Ingestion finished.')).not.toBeInTheDocument()
  })

  it('lists the PDFs an ingestion did not read as usual, one line per PDF', async () => {
    stubApi([
      {
        status: 'idle',
        last_error: null,
        pdf_problems: [
          { file: 'vejledning.pdf', problem: 'read on the CPU' },
          { file: 'vejledning.pdf', problem: 'almost no text (40 characters on 41 pages)' },
          { file: 'beskadiget.pdf', problem: 'skipped, nothing indexed (RuntimeError: not a PDF)' },
        ],
      },
    ])

    await startIngestion()

    expect(await screen.findByText('Ingestion finished, but 2 PDFs were not read as usual:')).toBeInTheDocument()
    const list = screen.getByRole('list', { name: 'PDFs not read as usual' })
    expect(within(list).getAllByRole('listitem').map((item) => item.textContent)).toEqual([
      'vejledning.pdf: read on the CPU; almost no text (40 characters on 41 pages)',
      'beskadiget.pdf: skipped, nothing indexed (RuntimeError: not a PDF)',
    ])
  })

  it('says only that it finished when an older server reports no outcome', async () => {
    stubApi([{ status: 'idle' }])

    await startIngestion()

    expect(await screen.findByText('Ingestion finished.')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})

describe('opening the panel', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('follows an ingestion that is still running', async () => {
    stubApi([
      { status: 'running', last_error: null, pdf_problems: [] },
      { status: 'idle', last_error: 'RuntimeError: embedding quota exceeded', pdf_problems: [] },
    ])

    render(<DocumentsPanel onClose={() => {}} />)

    expect(await screen.findByRole('button', { name: 'Ingestion running…' })).toBeDisabled()
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
    await act(() => vi.advanceTimersByTimeAsync(3000))
    expect(await screen.findByRole('alert')).toHaveTextContent('Ingestion failed: RuntimeError: embedding quota exceeded')
  })

  it('shows why the last ingestion failed', async () => {
    stubApi([{ status: 'idle', last_error: 'DoclingFailedError: docling could not read vejledning.pdf', pdf_problems: [] }])

    render(<DocumentsPanel onClose={() => {}} />)

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Ingestion failed: DoclingFailedError: docling could not read vejledning.pdf',
    )
  })

  it('says nothing about a last ingestion that went as usual', async () => {
    stubApi([{ status: 'idle', last_error: null, pdf_problems: [] }])

    render(<DocumentsPanel onClose={() => {}} />)
    await act(() => vi.advanceTimersByTimeAsync(3000))

    expect(screen.getByRole('button', { name: 'Start ingestion' })).toBeEnabled()
    expect(screen.queryByText(/Ingestion finished/)).not.toBeInTheDocument()
  })
})
