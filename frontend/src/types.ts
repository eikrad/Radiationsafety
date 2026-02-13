export interface SourceInfo {
  source: string
  document_type: string | null
}

export interface QueryResponse {
  answer: string
  sources: SourceInfo[]
  chat_history: [string, string][]  // [[q,a],[q,a],...]
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: SourceInfo[]
}
