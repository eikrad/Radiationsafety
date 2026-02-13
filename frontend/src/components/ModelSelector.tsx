import type { Model } from '../constants'

interface ModelSelectorProps {
  value: Model
  onChange: (model: Model) => void
}

const MODEL_LABELS: Record<Model, string> = {
  mistral: 'Mistral',
  gemini: 'Gemini',
  openai: 'OpenAI',
}

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
  return (
    <select
      className="model-selector"
      value={value}
      onChange={(e) => onChange(e.target.value as Model)}
      title="Select LLM provider"
    >
      {(Object.entries(MODEL_LABELS) as [Model, string][]).map(([id, label]) => (
        <option key={id} value={id}>
          {label}
        </option>
      ))}
    </select>
  )
}
