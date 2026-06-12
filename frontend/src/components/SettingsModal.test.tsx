import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { SettingsModal } from './SettingsModal'

const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = String(value)
    },
    removeItem: (key: string) => {
      delete store[key]
    },
    clear: () => {
      store = {}
    },
  }
})()

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
})

describe('SettingsModal - Privacy Mode', () => {
  beforeEach(() => {
    localStorageMock.clear()
  })

  it('renders privacy mode toggle checkbox', () => {
    render(<SettingsModal isOpen={true} onClose={vi.fn()} />)
    const privacyToggle = screen.getByRole('checkbox', { name: /privacy/i })
    expect(privacyToggle).toBeInTheDocument()
  })

  it('loads privacy mode state from localStorage on open', () => {
    localStorageMock.setItem('radiationsafety_enforce_privacy_mode', 'true')
    render(<SettingsModal isOpen={true} onClose={vi.fn()} />)
    const privacyToggle = screen.getByRole('checkbox', { name: /privacy/i }) as HTMLInputElement
    expect(privacyToggle.checked).toBe(true)
  })

  it('defaults to unchecked when localStorage is empty', () => {
    render(<SettingsModal isOpen={true} onClose={vi.fn()} />)
    const privacyToggle = screen.getByRole('checkbox', { name: /privacy/i }) as HTMLInputElement
    expect(privacyToggle.checked).toBe(false)
  })

  it('saves privacy mode state when toggled', async () => {
    const user = userEvent.setup()
    render(<SettingsModal isOpen={true} onClose={vi.fn()} />)
    const privacyToggle = screen.getByRole('checkbox', { name: /privacy/i })

    await user.click(privacyToggle)

    // Note: actual save happens in handleSave() when Close is clicked
    // This test verifies the checkbox state changes
    expect((privacyToggle as HTMLInputElement).checked).toBe(true)
  })

  it('persists privacy mode to localStorage on save', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(<SettingsModal isOpen={true} onClose={onClose} />)

    const privacyToggle = screen.getByRole('checkbox', { name: /privacy/i })
    await user.click(privacyToggle)

    // Click the "Save" button first to trigger handleSave
    const saveButton = screen.getByRole('button', { name: /Save/i })
    await user.click(saveButton)

    const stored = localStorageMock.getItem('radiationsafety_enforce_privacy_mode')
    expect(stored).toBe('true')
  })

  it('shows privacy mode hint text', () => {
    render(<SettingsModal isOpen={true} onClose={vi.fn()} />)
    expect(
      screen.getByText(/Run fully local with Ollama/i)
    ).toBeInTheDocument()
  })
})
