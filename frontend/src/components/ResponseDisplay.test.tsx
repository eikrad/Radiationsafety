import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ResponseDisplay } from './ResponseDisplay'

describe('ResponseDisplay', () => {
  it('returns null when no messages', () => {
    const { container } = render(<ResponseDisplay messages={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders user and assistant messages', () => {
    render(
      <ResponseDisplay
        messages={[
          { role: 'user', content: 'What is radiation?' },
          { role: 'assistant', content: 'Radiation is...' },
        ]}
      />
    )
    expect(screen.getByText('You')).toBeInTheDocument()
    expect(screen.getByText('AI Assistant')).toBeInTheDocument()
    expect(screen.getByText('What is radiation?')).toBeInTheDocument()
    expect(screen.getByText('Radiation is...')).toBeInTheDocument()
  })

  it('renders sources for assistant messages', () => {
    render(
      <ResponseDisplay
        messages={[
          { role: 'user', content: 'Q' },
          {
            role: 'assistant',
            content: 'A',
            sources: [
              { source: 'doc1.pdf', document_type: 'IAEA' },
              { source: 'doc2.pdf', document_type: null },
            ],
          },
        ]}
      />
    )
    expect(screen.getByText('Sources')).toBeInTheDocument()
    expect(screen.getByText('doc1.pdf')).toBeInTheDocument()
    expect(screen.getByText('doc2.pdf')).toBeInTheDocument()
    expect(screen.getByText(/IAEA/)).toBeInTheDocument()
  })

  it('renders warning when present', () => {
    render(
      <ResponseDisplay
        messages={[
          { role: 'user', content: 'Q' },
          {
            role: 'assistant',
            content: 'A',
            warning: 'Die Websuche konnte keine ausreichend guten Quellen liefern.',
          },
        ]}
      />
    )
    expect(screen.getByText(/Websuche.*Quellen/)).toBeInTheDocument()
  })

  it('renders multiple turns', () => {
    render(
      <ResponseDisplay
        messages={[
          { role: 'user', content: 'Q1' },
          { role: 'assistant', content: 'A1' },
          { role: 'user', content: 'And what about X?' },
          { role: 'assistant', content: 'X is...' },
        ]}
      />
    )
    expect(screen.getByText('Q1')).toBeInTheDocument()
    expect(screen.getByText('A1')).toBeInTheDocument()
    expect(screen.getByText('And what about X?')).toBeInTheDocument()
    expect(screen.getByText('X is...')).toBeInTheDocument()
  })
})

describe('ResponseDisplay while an answer is on its way', () => {
  it('shows the question at once, with progress in place of the answer', () => {
    render(<ResponseDisplay messages={[]} pendingQuestion="What is ALARA?" />)

    expect(screen.getByText('What is ALARA?')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent(/Searching/)
  })

  it('keeps earlier turns above the pending question', () => {
    render(
      <ResponseDisplay
        messages={[
          { role: 'user', content: 'Q1' },
          { role: 'assistant', content: 'A1' },
        ]}
        pendingQuestion="Q2"
      />
    )
    const texts = screen.getAllByText(/^(Q1|A1|Q2)$/).map((el) => el.textContent)
    expect(texts).toEqual(['Q1', 'A1', 'Q2'])
  })
})

describe('ResponseDisplay scrolling', () => {
  let scrolled: ReturnType<typeof vi.fn>

  beforeEach(() => {
    scrolled = vi.fn()
    // jsdom does not implement scrolling.
    Element.prototype.scrollIntoView = scrolled as unknown as Element['scrollIntoView']
  })

  it('brings a pending question into view', () => {
    const { rerender } = render(<ResponseDisplay messages={[{ role: 'user', content: 'Q1' }]} />)
    scrolled.mockClear()

    rerender(<ResponseDisplay messages={[{ role: 'user', content: 'Q1' }]} pendingQuestion="Q2" />)

    expect(scrolled).toHaveBeenCalled()
  })

  it('brings a new answer into view', () => {
    const { rerender } = render(<ResponseDisplay messages={[]} pendingQuestion="Q1" />)
    scrolled.mockClear()

    rerender(
      <ResponseDisplay
        messages={[
          { role: 'user', content: 'Q1' },
          { role: 'assistant', content: 'A1' },
        ]}
      />
    )

    expect(scrolled).toHaveBeenCalled()
  })
})
