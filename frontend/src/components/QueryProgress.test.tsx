import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryProgress } from './QueryProgress'

describe('QueryProgress', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('tells the user the documents are being searched', () => {
    render(<QueryProgress />)
    expect(screen.getByRole('status')).toHaveTextContent(/Searching the IAEA and Danish documents/)
  })

  it('shows how long the answer has been running', () => {
    render(<QueryProgress />)
    expect(screen.getByText('0 s')).toBeInTheDocument()

    act(() => {
      vi.advanceTimersByTime(12_000)
    })

    expect(screen.getByText('12 s')).toBeInTheDocument()
  })

  it('explains a long wait instead of leaving the user guessing', () => {
    render(<QueryProgress />)
    expect(screen.queryByText(/Still working/)).toBeNull()

    act(() => {
      vi.advanceTimersByTime(45_000)
    })

    expect(screen.getByText(/Still working/)).toBeInTheDocument()
  })
})
