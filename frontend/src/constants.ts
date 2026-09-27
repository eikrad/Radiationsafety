export const MODELS = ['scaleway', 'mistral', 'gemini', 'openai', 'ollama'] as const
export type Model = (typeof MODELS)[number]

/** Provider selected until the user picks another (EU-hosted; see backend DEFAULT_PROVIDER). */
export const DEFAULT_MODEL: Model = 'scaleway'

/** Base path for backend API (Vite dev proxy and production nginx use /api). */
export const API_BASE = '/api'

export const STORAGE_KEYS = {
  model: 'radiation-safety-model',
  apiKeys: 'radiation-safety-api-keys',
  modelVariants: 'radiation-safety-model-variants',
  /** When true, Documents panel shows "Search URL" (find document URL via web search). Beta. */
  documentSearchEnabled: 'radiation-safety-document-search-enabled',
  /** When true, questions may fall back to web search (if the server offers it). Off by default. */
  webSearchEnabled: 'radiation-safety-web-search-enabled',
  /** When true, restrict to Ollama only (privacy mode enforced). */
  enforcePrivacyMode: 'radiationsafety_enforce_privacy_mode',
} as const

/** Per-provider model variants. Key = provider, value = specific model ID. */
export const MODEL_VARIANTS: Record<Model, { id: string; label: string }[]> = {
  // Scaleway models come from GET /api/config (the server's .env), not from here.
  scaleway: [],
  mistral: [{ id: 'default', label: 'Mistral (default)' }],
  gemini: [
    { id: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro (default)' },
    { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash (10 RPM)' },
    { id: 'gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash-Lite (15 RPM, free tier)' },
  ],
  openai: [
    { id: 'gpt-4o-mini', label: 'GPT-4o mini (recommended)' },
    { id: 'gpt-4o', label: 'GPT-4o' },
  ],
  ollama: [{ id: 'default', label: 'Ollama (default)' }],
}

/** Readable names for Scaleway model ids; an id not listed here is shown as is. */
const SCALEWAY_MODEL_LABELS: Record<string, string> = {
  'gemma-4-26b-a4b-it': 'Gemma 4 26B',
  'deepseek-v4-flash-0731': 'DeepSeek V4 Flash',
  'qwen3.8-27b': 'Qwen3.8 27B',
}

export function scalewayModelLabel(id: string): string {
  return SCALEWAY_MODEL_LABELS[id] ?? id
}
