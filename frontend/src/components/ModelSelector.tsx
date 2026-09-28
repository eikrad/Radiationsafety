import { PROVIDER_LABELS, type Model } from '../constants'
import type { ProvidersStatus } from '../types'

interface ModelSelectorProps {
  value: Model
  onChange: (model: Model) => void
  enforcePrivacyMode?: boolean
  /** From GET /api/config; a provider with an issue is marked but stays selectable. */
  providers?: ProvidersStatus
}

export function ModelSelector({
  value,
  onChange,
  enforcePrivacyMode = false,
  providers = {},
}: ModelSelectorProps) {
  const activeIssue = providers[value]?.issue
  return (
    <select
      className={`model-selector${activeIssue ? ' model-selector--needs-setup' : ''}`}
      value={value}
      onChange={(e) => onChange(e.target.value as Model)}
      title={
        enforcePrivacyMode ? 'Privacy Mode: Ollama only' : (activeIssue ?? 'Select LLM provider')
      }
    >
      {(Object.entries(PROVIDER_LABELS) as [Model, string][]).map(([id, label]) => {
        const issue = providers[id]?.issue
        return (
          <option
            key={id}
            value={id}
            disabled={enforcePrivacyMode && id !== 'ollama'}
            title={issue ?? undefined}
          >
            {issue ? `${label} – needs setup` : label}
          </option>
        )
      })}
    </select>
  )
}
