import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

// jsdom here has no Web Storage; the app reads and writes it on start.
function memoryStorage(): Storage {
  let store: Record<string, string> = {}
  return {
    get length() {
      return Object.keys(store).length
    },
    key: (i) => Object.keys(store)[i] ?? null,
    getItem: (k) => store[k] ?? null,
    setItem: (k, v) => {
      store[k] = String(v)
    },
    removeItem: (k) => {
      delete store[k]
    },
    clear: () => {
      store = {}
    },
  }
}

/** A /query response the test resolves when it wants the answer to arrive. */
function deferredQuery() {
  let respond!: (body: unknown, status?: number) => void
  const response = new Promise<Response>((resolve) => {
    respond = (body, status = 200) =>
      resolve(new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }))
  })
  return { response, respond }
}

describe('asking a question', () => {
  let query: ReturnType<typeof deferredQuery>

  beforeEach(() => {
    vi.stubGlobal('localStorage', memoryStorage())
    vi.stubGlobal('sessionStorage', memoryStorage())
    query = deferredQuery()
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) =>
        url.endsWith('/query')
          ? query.response
          : Promise.resolve(new Response(JSON.stringify({ server_has_llm_key: true })))
      )
    )
    Element.prototype.scrollIntoView = vi.fn()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  async function ask(question: string) {
    const user = userEvent.setup()
    render(<App />)
    await user.type(screen.getByPlaceholderText(/Ask a question/i), question)
    await user.keyboard('{Enter}')
  }

  it('shows the question and progress while the answer is on its way', async () => {
    await ask('What is ALARA?')

    expect(screen.getByText('What is ALARA?')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent(/Searching/)
  })

  it('replaces the progress with the answer', async () => {
    await ask('What is ALARA?')

    query.respond({ answer: 'As low as reasonably achievable.', sources: [], chat_history: [] })

    expect(await screen.findByText('As low as reasonably achievable.')).toBeInTheDocument()
    expect(screen.queryByRole('status')).toBeNull()
    expect(screen.getAllByText('What is ALARA?')).toHaveLength(1)
  })

  it('removes the progress when the question fails', async () => {
    await ask('What is ALARA?')

    query.respond({ detail: 'Scaleway did not answer within 60 s.' }, 504)

    expect(await screen.findByText(/did not answer within 60 s/)).toBeInTheDocument()
    await waitFor(() => expect(screen.queryByRole('status')).toBeNull())
  })
})
