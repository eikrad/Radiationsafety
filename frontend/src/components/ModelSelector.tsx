import type { Model } from '../constants'

interface ModelSelectorProps {
  value: Model
  onChange: (model: Model) => void
  enforcePrivacyMode?: boolean
}

const MODEL_LABELS: Record<Model, string> = {
  mistral: 'Mistral',
  gemini: 'Gemini',
  openai: 'OpenAI',
  ollama: 'Ollama (Local)',
}

export function ModelSelector({ value, onChange, enforcePrivacyMode = false }: ModelSelectorProps) {
  return (
    <select
      className="model-selector"
      value={value}
      onChange={(e) => onChange(e.target.value as Model)}
      title={enforcePrivacyMode ? 'Privacy Mode: Ollama only' : 'Select LLM provider'}
    >
      {(Object.entries(MODEL_LABELS) as [Model, string][]).map(([id, label]) => (
        <option key={id} value={id} disabled={enforcePrivacyMode && id !== 'ollama'}>
          {label}
        </option>
      ))}
    </select>
  )
}
