import { PROVIDER_LABELS, type Model } from './constants'

export interface QueryErrorView {
  message: string
  /** Settings can fix it: a missing key, a rate limit, or a provider that needs setup. */
  openSettings: boolean
}

/** What to tell the user when POST /api/query fails; `body` is the parsed JSON, or null. */
export function describeQueryError(status: number, body: unknown, model: Model): QueryErrorView {
  const detail = (body as { detail?: unknown } | null)?.detail
  const message =
    typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? detail.map((e) => (e as { msg?: string }).msg ?? String(e)).join('; ')
        : fallbackMessage(status, model)
  const lower = message.toLowerCase()
  const openSettings =
    (status === 503 && model !== 'ollama') ||
    lower.includes('api key') ||
    lower.includes('rate limit') ||
    lower.includes('quota')
  return { message, openSettings }
}

function fallbackMessage(status: number, model: Model): string {
  if (status < 500) return `Unexpected response from server (HTTP ${status}).`
  if (model === 'ollama') {
    return `Server error (${status}). Check that the backend is running and Ollama is available.`
  }
  return (
    `Server error (${status}) while answering with ${PROVIDER_LABELS[model]}. ` +
    'Check the backend logs, or pick another provider.'
  )
}
