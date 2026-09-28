import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ModelSelector } from './ModelSelector'

describe('ModelSelector with Privacy Mode', () => {
  it('renders all models when privacy mode is off', () => {
    render(<ModelSelector value="mistral" onChange={vi.fn()} enforcePrivacyMode={false} />)
    const select = screen.getByRole('combobox') as HTMLSelectElement
    const options = Array.from(select.options).map((opt) => opt.value)
    expect(options).toContain('mistral')
    expect(options).toContain('gemini')
    expect(options).toContain('openai')
    expect(options).toContain('ollama')
  })

  it('disables non-ollama options when privacy mode is on', () => {
    render(<ModelSelector value="ollama" onChange={vi.fn()} enforcePrivacyMode={true} />)
    const select = screen.getByRole('combobox') as HTMLSelectElement
    const disabledOptions = Array.from(select.options).filter((opt) => opt.disabled)
    const disabledValues = disabledOptions.map((opt) => opt.value)
    expect(disabledValues).toContain('mistral')
    expect(disabledValues).toContain('gemini')
    expect(disabledValues).toContain('openai')
  })

  it('ollama option is always enabled', () => {
    render(<ModelSelector value="ollama" onChange={vi.fn()} enforcePrivacyMode={true} />)
    const select = screen.getByRole('combobox') as HTMLSelectElement
    const ollamaOption = Array.from(select.options).find((opt) => opt.value === 'ollama')
    expect(ollamaOption?.disabled).toBe(false)
  })

  it('allows selecting non-ollama when privacy mode is off', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<ModelSelector value="ollama" onChange={onChange} enforcePrivacyMode={false} />)
    const select = screen.getByRole('combobox')
    await user.selectOptions(select, 'mistral')
    expect(onChange).toHaveBeenCalledWith('mistral')
  })
})

describe('ModelSelector providers', () => {
  it('offers Scaleway first, as the default provider', () => {
    render(<ModelSelector value="scaleway" onChange={vi.fn()} />)
    const select = screen.getByRole('combobox') as HTMLSelectElement
    expect(select.options[0].value).toBe('scaleway')
    expect(select.options[0].textContent).toMatch(/Scaleway/)
    expect(Array.from(select.options).map((o) => o.value)).toContain('gemini')
  })
})

describe('ModelSelector provider status', () => {
  const needsSetup = {
    scaleway: { server_key: true, issue: 'Scaleway has no answer model on this server: set SCW_MODEL.' },
    gemini: { server_key: true, issue: null },
  }

  it('marks a provider the server cannot use yet, but keeps it selectable', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<ModelSelector value="gemini" onChange={onChange} providers={needsSetup} />)
    const select = screen.getByRole('combobox') as HTMLSelectElement
    const scaleway = Array.from(select.options).find((o) => o.value === 'scaleway')!

    expect(scaleway.textContent).toBe('Scaleway (EU) – needs setup')
    expect(scaleway.title).toMatch(/SCW_MODEL/)
    await user.selectOptions(select, 'scaleway')
    expect(onChange).toHaveBeenCalledWith('scaleway')
  })

  it('leaves ready providers unmarked', () => {
    render(<ModelSelector value="gemini" onChange={vi.fn()} providers={needsSetup} />)
    expect(screen.getByRole('option', { name: 'Gemini' })).toBeInTheDocument()
  })
})
