import { useState, useEffect } from 'react'
import {
  MODELS,
  MODEL_VARIANTS,
  PROVIDER_LABELS,
  STORAGE_KEYS,
  scalewayModelLabel,
  type Model,
} from '../constants'
import {
  loadApiKeys,
  saveApiKeys,
  loadDocumentSearchEnabled,
  loadModelVariants,
  loadEnforcePrivacyMode,
  saveEnforcePrivacyMode,
} from '../storage'
import type { ProviderStatus, ProvidersStatus } from '../types'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
  /** From GET /api/config: the server's default Scaleway model first, then the allowed ones. */
  scalewayModels?: string[]
  /** The provider picked in the header; its section is listed first. */
  activeModel?: Model
  /** From GET /api/config: per provider, server key and blocking issue. */
  providers?: ProvidersStatus
}

/** Model choices per provider; Scaleway's come from the server, the default first. */
function variantOptions(model: Model, scalewayModels: string[]): { id: string; label: string }[] {
  if (model !== 'scaleway') return MODEL_VARIANTS[model]
  return scalewayModels.map((id, i) =>
    i === 0
      ? { id: 'default', label: `${scalewayModelLabel(id)} (default)` }
      : { id, label: scalewayModelLabel(id) }
  )
}

type ProviderState = 'ready' | 'needs-setup' | 'key-required' | 'unknown'

const STATE_LABELS: Record<Exclude<ProviderState, 'unknown'>, string> = {
  ready: 'Ready',
  'needs-setup': 'Needs setup',
  'key-required': 'Key required',
}

/** Whether the provider can answer: server config first, then a key from server or browser. */
function providerState(model: Model, status: ProviderStatus | undefined, browserKey: string): ProviderState {
  if (!status) return 'unknown'
  if (status.issue) return 'needs-setup'
  if (model === 'ollama' || status.server_key || browserKey.trim()) return 'ready'
  return 'key-required'
}

export function SettingsModal({
  isOpen,
  onClose,
  scalewayModels = [],
  activeModel,
  providers = {},
}: SettingsModalProps) {
  const [keys, setKeys] = useState<Record<Model, string>>(loadApiKeys())
  const [variants, setVariants] = useState<Record<Model, string>>(loadModelVariants())
  const [showKeys, setShowKeys] = useState<Record<Model, boolean>>({
    scaleway: false,
    mistral: false,
    gemini: false,
    openai: false,
    ollama: false,
  })
  const [documentSearchEnabled, setDocumentSearchEnabled] = useState(loadDocumentSearchEnabled())
  const [enforcePrivacyMode, setEnforcePrivacyMode] = useState(loadEnforcePrivacyMode())

  useEffect(() => {
    if (!isOpen) return
    queueMicrotask(() => {
      setKeys(loadApiKeys())
      setVariants(loadModelVariants())
      setDocumentSearchEnabled(loadDocumentSearchEnabled())
      setEnforcePrivacyMode(loadEnforcePrivacyMode())
    })
  }, [isOpen])

  function handleChange(model: Model, value: string) {
    setKeys((prev) => ({ ...prev, [model]: value }))
  }

  function handleToggleShow(model: Model) {
    setShowKeys((prev) => ({ ...prev, [model]: !prev[model] }))
  }

  function handleVariantChange(model: Model, value: string) {
    setVariants((prev) => ({ ...prev, [model]: value }))
  }

  function handleSave() {
    try {
      saveApiKeys(keys)
      localStorage.setItem(STORAGE_KEYS.modelVariants, JSON.stringify(variants))
      localStorage.setItem(STORAGE_KEYS.documentSearchEnabled, String(documentSearchEnabled))
      saveEnforcePrivacyMode(enforcePrivacyMode)
      onClose()
    } catch (e) {
      console.error('Failed to save settings:', e)
    }
  }

  if (!isOpen) return null

  const orderedModels = activeModel
    ? [activeModel, ...MODELS.filter((m) => m !== activeModel)]
    : [...MODELS]

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>Settings</h2>
          <button type="button" className="settings-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <section className="settings-section" aria-labelledby="settings-general-heading">
          <h3 id="settings-general-heading" className="settings-section-title">
            General
          </h3>
          <div className="settings-field">
            <label className="settings-toggle-label">
              <input
                type="checkbox"
                checked={enforcePrivacyMode}
                onChange={(e) => setEnforcePrivacyMode(e.target.checked)}
                aria-describedby="privacy-mode-desc"
              />
              <span>🔒 Privacy Mode</span>
            </label>
            <p id="privacy-mode-desc" className="settings-field-desc">
              Run fully local with Ollama. No API keys required. No data leaves your machine.
            </p>
          </div>
          <div className="settings-field">
            <label className="settings-toggle-label">
              <input
                type="checkbox"
                checked={documentSearchEnabled}
                onChange={(e) => setDocumentSearchEnabled(e.target.checked)}
                aria-describedby="document-search-desc"
              />
              <span>
                Search for new documents <span className="settings-beta-badge" aria-hidden>Beta</span>
              </span>
            </label>
            <p id="document-search-desc" className="settings-field-desc">
              When enabled, the Documents panel shows a “Search URL” action to find document URLs via
              web search (IAEA or retsinformation.dk). This feature is experimental.
            </p>
          </div>
        </section>
        <section className="settings-section" aria-labelledby="settings-providers-heading">
          <h3 id="settings-providers-heading" className="settings-section-title">
            Providers
          </h3>
          <p className="settings-hint">
            Pick the provider in the header. Keys entered here are stored only in this browser tab,
            sent only with your questions, and cleared when you close the tab.
          </p>
          <div className="settings-provider-list">
            {orderedModels.map((model) => {
              const status = providers[model]
              const state = providerState(model, status, keys[model])
              const isActive = model === activeModel
              const options = variantOptions(model, scalewayModels)
              const lockedByPrivacy = enforcePrivacyMode && model !== 'ollama'
              return (
                <section
                  key={model}
                  aria-label={PROVIDER_LABELS[model]}
                  className={[
                    'settings-provider',
                    isActive && 'settings-provider--active',
                    lockedByPrivacy && 'settings-provider--locked',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                >
                  <div className="settings-provider-header">
                    <h4>{PROVIDER_LABELS[model]}</h4>
                    {isActive && <span className="settings-badge settings-badge--active">Active</span>}
                    {state !== 'unknown' && (
                      <span className={`settings-badge settings-badge--${state}`}>
                        {STATE_LABELS[state]}
                      </span>
                    )}
                  </div>
                  {status?.issue && (
                    <p className="settings-provider-issue" role="note">
                      {status.issue}
                    </p>
                  )}
                  {lockedByPrivacy && (
                    <p className="settings-field-desc">Not used while Privacy Mode is on.</p>
                  )}
                  {model === 'ollama' ? (
                    <p className="settings-field-desc">
                      Runs fully local; no API key needed and no data leaves your machine. Model
                      and URL are set on the server.
                    </p>
                  ) : (
                    <div className="settings-field">
                      <label htmlFor={`api-key-${model}`}>API key</label>
                      <div className="settings-input-row">
                        <input
                          id={`api-key-${model}`}
                          type={showKeys[model] ? 'text' : 'password'}
                          value={keys[model]}
                          onChange={(e) => handleChange(model, e.target.value)}
                          placeholder={status?.server_key ? 'Optional' : 'Enter API key...'}
                          autoComplete="off"
                          aria-describedby={status?.server_key ? `api-key-${model}-desc` : undefined}
                        />
                        <button
                          type="button"
                          className="settings-toggle-visibility"
                          onClick={() => handleToggleShow(model)}
                          aria-label={showKeys[model] ? 'Hide' : 'Show'}
                          title={showKeys[model] ? 'Hide' : 'Show'}
                        >
                          {showKeys[model] ? 'Hide' : 'Show'}
                        </button>
                      </div>
                      {status?.server_key && (
                        <p id={`api-key-${model}-desc`} className="settings-field-desc">
                          Optional: the server already has a key. One entered here is used instead.
                        </p>
                      )}
                    </div>
                  )}
                  {options.length > 1 && (
                    <div className="settings-field">
                      <label htmlFor={`variant-${model}`}>Model</label>
                      <select
                        className="settings-select"
                        id={`variant-${model}`}
                        value={variants[model]}
                        onChange={(e) => handleVariantChange(model, e.target.value)}
                      >
                        {options.map((v) => (
                          <option key={v.id} value={v.id}>
                            {v.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                  {options.length === 1 && model === 'scaleway' && (
                    <p className="settings-field-desc">Model: {scalewayModelLabel(scalewayModels[0]!)}</p>
                  )}
                </section>
              )
            })}
          </div>
        </section>
        <div className="settings-actions">
          <button type="button" className="settings-save" onClick={handleSave}>
            Save
          </button>
          <button type="button" className="settings-cancel" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
