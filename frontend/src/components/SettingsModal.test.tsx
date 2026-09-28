import { render, screen, within } from '@testing-library/react'
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

describe('SettingsModal - Scaleway', () => {
  beforeEach(() => {
    localStorageMock.clear()
  })

  it('has a Scaleway API key field', () => {
    render(<SettingsModal isOpen={true} onClose={vi.fn()} />)
    const scaleway = screen.getByRole('region', { name: 'Scaleway (EU)' })
    expect(within(scaleway).getByLabelText('API key')).toBeInTheDocument()
  })

  it('offers the Scaleway models the server allows, by readable name', () => {
    render(
      <SettingsModal
        isOpen={true}
        onClose={vi.fn()}
        scalewayModels={['gemma-4-26b-a4b-it', 'deepseek-v4-flash-0731', 'some-new-model']}
      />
    )
    const select = document.getElementById('variant-scaleway') as HTMLSelectElement
    const labels = Array.from(select.options).map((o) => o.textContent)
    expect(labels).toEqual(['Gemma 4 26B (default)', 'DeepSeek V4 Flash', 'some-new-model'])
    expect(select.value).toBe('default')
  })
})

describe('SettingsModal - providers', () => {
  beforeEach(() => {
    localStorageMock.clear()
  })

  function providerSections() {
    const list = screen.getByRole('region', { name: 'Providers' })
    return within(list)
      .getAllByRole('region')
      .map((r) => r.getAttribute('aria-label'))
  }

  it('shows each provider in its own section', () => {
    render(<SettingsModal isOpen={true} onClose={vi.fn()} />)
    expect(providerSections()).toEqual(
      expect.arrayContaining(['Scaleway (EU)', 'Mistral', 'Gemini', 'OpenAI', 'Ollama (Local)'])
    )
  })

  it('lists the active provider first and marks it as active', () => {
    render(<SettingsModal isOpen={true} onClose={vi.fn()} activeModel="gemini" />)
    expect(providerSections()[0]).toBe('Gemini')
    expect(within(screen.getByRole('region', { name: 'Gemini' })).getByText('Active')).toBeInTheDocument()
    expect(within(screen.getByRole('region', { name: 'OpenAI' })).queryByText('Active')).toBeNull()
  })

  it('explains what the server is missing for a provider', () => {
    render(
      <SettingsModal
        isOpen={true}
        onClose={vi.fn()}
        providers={{ scaleway: { server_key: true, issue: 'Scaleway has no answer model on this server: set SCW_MODEL.' } }}
      />
    )
    const scaleway = screen.getByRole('region', { name: 'Scaleway (EU)' })
    expect(within(scaleway).getByText('Needs setup')).toBeInTheDocument()
    expect(within(scaleway).getByText(/set SCW_MODEL/)).toBeInTheDocument()
  })

  it('says a browser key is optional when the server holds one', () => {
    render(
      <SettingsModal
        isOpen={true}
        onClose={vi.fn()}
        providers={{
          gemini: { server_key: true, issue: null },
          openai: { server_key: false, issue: null },
        }}
      />
    )
    expect(within(screen.getByRole('region', { name: 'Gemini' })).getByText('Ready')).toBeInTheDocument()
    expect(
      within(screen.getByRole('region', { name: 'Gemini' })).getByText(/optional/i)
    ).toBeInTheDocument()
    expect(
      within(screen.getByRole('region', { name: 'OpenAI' })).getByText('Key required')
    ).toBeInTheDocument()
  })
})
