import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { QueryForm } from './QueryForm'

describe('QueryForm', () => {
  it('renders textarea and submit button', () => {
    render(<QueryForm onSubmit={vi.fn()} loading={false} disabled={false} />)
    expect(screen.getByPlaceholderText(/Ask a question/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Ask/i })).toBeInTheDocument()
  })

  it('shows Searching when loading', () => {
    render(<QueryForm onSubmit={vi.fn()} loading={true} disabled={false} />)
    expect(screen.getByRole('button', { name: /Searching/i })).toBeInTheDocument()
  })

  it('calls onSubmit with question and clears field when form submitted', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(<QueryForm onSubmit={onSubmit} loading={false} disabled={false} />)
    const textarea = screen.getByPlaceholderText(/Ask a question/i)
    await user.type(textarea, 'What is radiation?')
    await user.click(screen.getByRole('button', { name: /Ask/i }))
    expect(onSubmit).toHaveBeenCalledWith('What is radiation?')
    expect(textarea).toHaveValue('')
  })

  it('does not call onSubmit when question is empty', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(<QueryForm onSubmit={onSubmit} loading={false} disabled={false} />)
    await user.click(screen.getByRole('button', { name: /Ask/i }))
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it('submits on Enter key', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(<QueryForm onSubmit={onSubmit} loading={false} disabled={false} />)
    const textarea = screen.getByPlaceholderText(/Ask a question/i)
    await user.type(textarea, 'What is ALARA?')
    await user.keyboard('{Enter}')
    expect(onSubmit).toHaveBeenCalledWith('What is ALARA?')
  })

  it('inserts newline on Shift+Enter and does not submit', async () => {
    const onSubmit = vi.fn()
    const user = userEvent.setup()
    render(<QueryForm onSubmit={onSubmit} loading={false} disabled={false} />)
    const textarea = screen.getByPlaceholderText(/Ask a question/i)
    await user.type(textarea, 'line one')
    await user.keyboard('{Shift>}{Enter}{/Shift}')
    expect(onSubmit).not.toHaveBeenCalled()
    expect(textarea).toHaveValue('line one\n')
  })
})
