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
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: SourceInfo[]
  warning?: string | null
  used_web_search?: boolean
  used_web_search_label?: string | null
}
