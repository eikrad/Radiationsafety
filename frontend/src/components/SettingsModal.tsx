import { useState, useEffect } from 'react'
import { MODELS, MODEL_VARIANTS, STORAGE_KEYS, type Model } from '../constants'
import { loadApiKeys, loadModelVariants } from '../storage'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
}

const MODEL_LABELS: Record<Model, string> = {
  mistral: 'Mistral API Key',
  gemini: 'Gemini API Key',
  openai: 'OpenAI API Key',
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [keys, setKeys] = useState<Record<Model, string>>(loadApiKeys())
  const [variants, setVariants] = useState<Record<Model, string>>(loadModelVariants())
  const [showKeys, setShowKeys] = useState<Record<Model, boolean>>({
    mistral: false,
    gemini: false,
    openai: false,
  })

  useEffect(() => {
    if (isOpen) {
      setKeys(loadApiKeys())
      setVariants(loadModelVariants())
    }
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
      localStorage.setItem(STORAGE_KEYS.apiKeys, JSON.stringify(keys))
      localStorage.setItem(STORAGE_KEYS.modelVariants, JSON.stringify(variants))
      onClose()
    } catch (e) {
      console.error('Failed to save settings:', e)
    }
  }

  if (!isOpen) return null

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>Settings</h2>
          <button type="button" className="settings-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <p className="settings-hint">
          Keys are stored only in your browser and are never sent to our servers except for LLM API
          calls.
        </p>
        <div className="settings-fields">
          {MODELS.map((model) => (
            <div key={model} className="settings-field-block">
              <div className="settings-field">
                <label htmlFor={`api-key-${model}`}>{MODEL_LABELS[model]}</label>
                <div className="settings-input-row">
                  <input
                    id={`api-key-${model}`}
                    type={showKeys[model] ? 'text' : 'password'}
                    value={keys[model]}
                    onChange={(e) => handleChange(model, e.target.value)}
                    placeholder="Enter API key..."
                    autoComplete="off"
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
              </div>
              {MODEL_VARIANTS[model].length > 1 && (
                <div className="settings-field settings-model-variant">
                  <label htmlFor={`variant-${model}`}>Model</label>
                  <select
                    id={`variant-${model}`}
                    value={variants[model]}
                    onChange={(e) => handleVariantChange(model, e.target.value)}
                  >
                    {MODEL_VARIANTS[model].map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.label}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          ))}
        </div>
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
