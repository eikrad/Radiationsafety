import { STORAGE_KEYS, type Model } from './constants'

export function loadDocumentSearchEnabled(): boolean {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.documentSearchEnabled)
    if (raw === null) return false
    return raw === 'true'
  } catch {
    return false
  }
}

const DEFAULT_VARIANTS: Record<Model, string> = {
  mistral: 'default',
  gemini: 'gemini-2.5-flash-lite',
  openai: 'gpt-4o-mini',
}

export function loadApiKeys(): Record<Model, string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.apiKeys)
    if (!raw) return { mistral: '', gemini: '', openai: '' }
    const parsed = JSON.parse(raw) as Record<string, string>
    return {
      mistral: parsed.mistral ?? '',
      gemini: parsed.gemini ?? '',
      openai: parsed.openai ?? '',
    }
  } catch {
    return { mistral: '', gemini: '', openai: '' }
  }
}

/** True if at least one provider has a non-empty key in the UI (localStorage). */
export function hasAnyApiKeyInStorage(): boolean {
  const keys = loadApiKeys()
  return keys.mistral !== '' || keys.gemini !== '' || keys.openai !== ''
}

export function loadModelVariants(): Record<Model, string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.modelVariants)
    if (!raw) return { ...DEFAULT_VARIANTS }
    const parsed = JSON.parse(raw) as Record<string, string>
    return {
      mistral: parsed.mistral ?? DEFAULT_VARIANTS.mistral,
      gemini: parsed.gemini ?? DEFAULT_VARIANTS.gemini,
      openai: parsed.openai ?? DEFAULT_VARIANTS.openai,
    }
  } catch {
    return { ...DEFAULT_VARIANTS }
  }
}
