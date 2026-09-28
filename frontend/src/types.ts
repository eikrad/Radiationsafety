export interface SourceInfo {
  source: string
  document_type: string | null
}

export interface QueryResponse {
  answer: string
  sources: SourceInfo[]
  chat_history: [string, string][]  // [[q,a],[q,a],...]
  warning?: string | null
  used_web_search?: boolean
  used_web_search_label?: string | null  // in question's language
  privacy_mode?: boolean
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: SourceInfo[]
  warning?: string | null
  used_web_search?: boolean
  used_web_search_label?: string | null
}

/** From GET /api/config, per provider: can the server answer with it? */
export interface ProviderStatus {
  /** The server holds this provider's API key, so none is needed in Settings. */
  server_key: boolean
  /** Server configuration that stops the provider from answering, else null. */
  issue: string | null
}

export type ProvidersStatus = Partial<Record<string, ProviderStatus>>
