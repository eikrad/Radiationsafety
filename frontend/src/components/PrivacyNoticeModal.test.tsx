import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { PrivacyNoticeModal } from './PrivacyNoticeModal'

describe('PrivacyNoticeModal', () => {
  it('renders nothing when closed', () => {
    const { container } = render(
      <PrivacyNoticeModal
        isOpen={false}
        onClose={vi.fn()}
        controllerName={null}
        controllerContact={null}
      />
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders the notice heading and key sections when open', () => {
    render(
      <PrivacyNoticeModal
        isOpen={true}
        onClose={vi.fn()}
        controllerName={null}
        controllerContact={null}
      />
    )
    expect(screen.getByRole('heading', { name: 'Privacy notice' })).toBeInTheDocument()
    expect(screen.getByText(/not legal or clinical advice/i)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /Your rights/i })).toBeInTheDocument()
    expect(screen.getByText(/patient-identifiable/i)).toBeInTheDocument()
  })

  it('explains the solo-local fallback when no controller is configured', () => {
    render(
      <PrivacyNoticeModal
        isOpen={true}
        onClose={vi.fn()}
        controllerName={null}
        controllerContact={null}
      />
    )
    expect(screen.getByText(/running locally for your own use/i)).toBeInTheDocument()
    expect(screen.queryByText(/^Controller:/)).not.toBeInTheDocument()
  })

  it('names the configured controller and contact when set', () => {
    render(
      <PrivacyNoticeModal
        isOpen={true}
        onClose={vi.fn()}
        controllerName="Acme Hospital Physics Dept."
        controllerContact="privacy@acme.example"
      />
    )
    expect(screen.getByText(/Acme Hospital Physics Dept\./)).toBeInTheDocument()
    expect(screen.getByText(/privacy@acme\.example/)).toBeInTheDocument()
    expect(screen.queryByText(/running locally for your own use/i)).not.toBeInTheDocument()
  })

  it('names the configured controller even without a contact', () => {
    render(
      <PrivacyNoticeModal
        isOpen={true}
        onClose={vi.fn()}
        controllerName="Acme Hospital Physics Dept."
        controllerContact={null}
      />
    )
    expect(screen.getByText(/Acme Hospital Physics Dept\./)).toBeInTheDocument()
    expect(screen.queryByText(/Contact:/)).not.toBeInTheDocument()
  })

  it('calls onClose when the close button is clicked', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(
      <PrivacyNoticeModal
        isOpen={true}
        onClose={onClose}
        controllerName={null}
        controllerContact={null}
      />
    )
    await user.click(screen.getByRole('button', { name: 'Close' }))
    expect(onClose).toHaveBeenCalled()
  })
})
