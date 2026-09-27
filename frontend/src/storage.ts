import { STORAGE_KEYS, type Model } from './constants'

export function loadEnforcePrivacyMode(): boolean {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.enforcePrivacyMode)
    if (raw === null) return false
    return raw === 'true'
  } catch {
    return false
  }
}

export function saveEnforcePrivacyMode(enabled: boolean): void {
  try {
    localStorage.setItem(STORAGE_KEYS.enforcePrivacyMode, String(enabled))
  } catch {
    // Silently fail if localStorage is unavailable
  }
}

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
  // 'default' = the server's SCW_MODEL
  scaleway: 'default',
  mistral: 'default',
  gemini: 'gemini-2.5-pro',
  openai: 'gpt-4o-mini',
}

// API keys use sessionStorage, not localStorage: they must not outlive the tab.
// sessionStorage is cleared by the browser itself when the tab closes, which
// covers the crash/force-kill/mobile-backgrounding cases that the beforeunload/
// pagehide handlers in App.tsx can miss.
export function loadApiKeys(): Record<Model, string> {
  // Earlier versions kept keys in localStorage; drop any leftover (e.g. after a
  // crash skipped the unload cleanup) rather than reusing it.
  try {
    localStorage.removeItem(STORAGE_KEYS.apiKeys)
  } catch {
    // localStorage unavailable: nothing to clean up
  }
  try {
    const raw = sessionStorage.getItem(STORAGE_KEYS.apiKeys)
    if (!raw) return { scaleway: '', mistral: '', gemini: '', openai: '' }
    const parsed = JSON.parse(raw) as Record<string, string>
    return {
      scaleway: parsed.scaleway ?? '',
      mistral: parsed.mistral ?? '',
      gemini: parsed.gemini ?? '',
      openai: parsed.openai ?? '',
    }
  } catch {
    return { scaleway: '', mistral: '', gemini: '', openai: '' }
  }
}

export function saveApiKeys(keys: Record<Model, string>): void {
  try {
    sessionStorage.setItem(STORAGE_KEYS.apiKeys, JSON.stringify(keys))
  } catch {
    // Silently fail if sessionStorage is unavailable
  }
}

/** True if at least one provider has a non-empty key in the UI (sessionStorage). */
export function hasAnyApiKeyInStorage(): boolean {
  const keys = loadApiKeys()
  return Object.values(keys).some((key) => key !== '')
}

export function loadModelVariants(): Record<Model, string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.modelVariants)
    if (!raw) return { ...DEFAULT_VARIANTS }
    const parsed = JSON.parse(raw) as Record<string, string>
    return {
      scaleway: parsed.scaleway ?? DEFAULT_VARIANTS.scaleway,
      mistral: parsed.mistral ?? DEFAULT_VARIANTS.mistral,
      gemini: parsed.gemini ?? DEFAULT_VARIANTS.gemini,
      openai: parsed.openai ?? DEFAULT_VARIANTS.openai,
    }
  } catch {
    return { ...DEFAULT_VARIANTS }
  }
}
