import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { PrivacyNoticeModal } from './PrivacyNoticeModal'

describe('PrivacyNoticeModal', () => {
  it('renders nothing when closed', () => {
    const { container } = render(
      <PrivacyNoticeModal isOpen={false} onClose={vi.fn()} />
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders the notice heading and key sections when open', () => {
    render(<PrivacyNoticeModal isOpen={true} onClose={vi.fn()} />)
    expect(screen.getByRole('heading', { name: 'Privacy notice' })).toBeInTheDocument()
    expect(screen.getByText(/not legal or clinical advice/i)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /Your rights/i })).toBeInTheDocument()
    expect(screen.getByText(/patient-identifiable/i)).toBeInTheDocument()
  })

  it('calls onClose when the close button is clicked', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(<PrivacyNoticeModal isOpen={true} onClose={onClose} />)
    await user.click(screen.getByRole('button', { name: 'Close' }))
    expect(onClose).toHaveBeenCalled()
  })
})
