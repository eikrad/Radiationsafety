import { STORAGE_KEYS, type Model } from './constants'

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
