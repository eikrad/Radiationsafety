export const MODELS = ['mistral', 'gemini', 'openai'] as const
export type Model = (typeof MODELS)[number]

export const STORAGE_KEYS = {
  model: 'radiation-safety-model',
  apiKeys: 'radiation-safety-api-keys',
  modelVariants: 'radiation-safety-model-variants',
} as const

/** Per-provider model variants. Key = provider, value = specific model ID. */
export const MODEL_VARIANTS: Record<Model, { id: string; label: string }[]> = {
  mistral: [{ id: 'default', label: 'Mistral (default)' }],
  gemini: [
    { id: 'gemini-2.5-flash-lite', label: 'Gemini 2.5 Flash-Lite (15 RPM, recommended)' },
    { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash (10 RPM)' },
    { id: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro (5 RPM)' },
  ],
  openai: [
    { id: 'gpt-4o-mini', label: 'GPT-4o mini (recommended)' },
    { id: 'gpt-4o', label: 'GPT-4o' },
  ],
}
