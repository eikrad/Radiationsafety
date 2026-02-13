import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { ResponseDisplay } from './ResponseDisplay'

describe('ResponseDisplay', () => {
  it('returns null when no answer and no sources', () => {
    const { container } = render(<ResponseDisplay answer="" sources={[]} />)
    expect(container.firstChild).toBeNull()
  })

  it('renders answer when provided', () => {
    render(<ResponseDisplay answer="Radiation is..." sources={[]} />)
    expect(screen.getByText('Answer')).toBeInTheDocument()
    expect(screen.getByText('Radiation is...')).toBeInTheDocument()
  })

  it('renders sources when provided', () => {
    render(
      <ResponseDisplay
        answer=""
        sources={[
          { source: 'doc1.pdf', document_type: 'IAEA' },
          { source: 'doc2.pdf', document_type: null },
        ]}
      />
    )
    expect(screen.getByText('Sources')).toBeInTheDocument()
    expect(screen.getByText('doc1.pdf')).toBeInTheDocument()
    expect(screen.getByText('doc2.pdf')).toBeInTheDocument()
    expect(screen.getByText(/\IAEA/)).toBeInTheDocument()
  })

  it('renders both answer and sources', () => {
    render(
      <ResponseDisplay
        answer="The answer"
        sources={[{ source: 'ref.pdf', document_type: 'Danish law' }]}
      />
    )
    expect(screen.getByText('The answer')).toBeInTheDocument()
    expect(screen.getByText('ref.pdf')).toBeInTheDocument()
    expect(screen.getByText(/Danish law/)).toBeInTheDocument()
  })
})
