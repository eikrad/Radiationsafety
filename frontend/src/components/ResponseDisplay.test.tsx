import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
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
    expect(screen.getByText('Assistant')).toBeInTheDocument()
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
