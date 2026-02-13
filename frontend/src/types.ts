export interface SourceInfo {
  source: string
  document_type: string | null
}

export interface QueryResponse {
  answer: string
  sources: SourceInfo[]
}
