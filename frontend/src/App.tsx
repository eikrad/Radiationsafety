import { useEffect, useState } from 'react'
import { DocumentsPanel } from './components/DocumentsPanel'
import { ModelSelector } from './components/ModelSelector'
import { QueryForm } from './components/QueryForm'
import { ResponseDisplay } from './components/ResponseDisplay'
import { SettingsModal } from './components/SettingsModal'
import { MODELS, STORAGE_KEYS, type Model } from './constants'
import { loadApiKeys, loadModelVariants, hasAnyApiKeyInStorage } from './storage'
import type { Message, QueryResponse } from './types'
import './App.css'

const API_BASE = '/api'

function loadStoredModel(): Model {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.model)
    if (raw && MODELS.includes(raw as Model)) return raw as Model
  } catch {}
  return 'mistral'
}

export default function App() {
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [error, setError] = useState('')
  const [model, setModel] = useState<Model>(loadStoredModel)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [documentsOpen, setDocumentsOpen] = useState(false)
  /** From GET /api/config: true = server has .env keys (hide hint), false = needs key from client or .env. null = not yet loaded. */
  const [serverHasLlmKey, setServerHasLlmKey] = useState<boolean | null>(null)

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEYS.model, model)
    } catch {}
  }, [model])

  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE}/config`)
      .then((res) => res.json())
      .then((data: { server_has_llm_key?: boolean }) => {
        if (!cancelled) setServerHasLlmKey(Boolean(data.server_has_llm_key))
      })
      .catch(() => {
        if (!cancelled) setServerHasLlmKey(false)
      })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    function clearApiKeysOnClose() {
      try {
        localStorage.removeItem(STORAGE_KEYS.apiKeys)
      } catch {}
    }
    window.addEventListener('beforeunload', clearApiKeysOnClose)
    window.addEventListener('pagehide', clearApiKeysOnClose)
    return () => {
      window.removeEventListener('beforeunload', clearApiKeysOnClose)
      window.removeEventListener('pagehide', clearApiKeysOnClose)
    }
  }, [])

  async function handleSubmit(question: string) {
    setLoading(true)
    setError('')
    const chatHistory: [string, string][] = []
    for (let i = 0; i < messages.length - 1; i++) {
      if (messages[i]?.role === 'user' && messages[i + 1]?.role === 'assistant') {
        chatHistory.push([messages[i]!.content, messages[i + 1]!.content])
      }
    }
    const apiKeys = loadApiKeys()
    const variants = loadModelVariants()
    const payload: Record<string, unknown> = {
      question,
      chat_history: chatHistory,
      model,
      model_variant: variants[model] !== 'default' ? variants[model] : undefined,
    }
    if (apiKeys[model]) {
      payload.api_keys = { [model]: apiKeys[model] }
    }
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = (await res.json()) as QueryResponse & { detail?: string | unknown }
      if (!res.ok) {
        const detail = data.detail
        const msg =
          typeof detail === 'string'
            ? detail
            : Array.isArray(detail)
              ? detail.map((e) => (e as { msg?: string }).msg ?? String(e)).join('; ')
              : 'Request failed'
        throw new Error(msg)
      }
      const newMessages: Message[] = [
        ...messages,
        { role: 'user', content: question },
        {
          role: 'assistant',
          content: data.answer ?? '',
          sources: data.sources ?? [],
          warning: data.warning ?? null,
          used_web_search: data.used_web_search ?? false,
          used_web_search_label: data.used_web_search_label ?? null,
        },
      ]
      setMessages(newMessages)
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : 'Failed to get answer. Is the backend running?'
      setError(msg)
      const msgLower = msg.toLowerCase()
      if (msgLower.includes('api key') || msgLower.includes('rate limit') || msgLower.includes('quota')) {
        setSettingsOpen(true)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      {documentsOpen && (
        <DocumentsPanel onClose={() => setDocumentsOpen(false)} />
      )}
      <div className="app-body">
        <header className="app-header">
          <div className="header-left">
            <button
              type="button"
              className="header-btn"
              onClick={() => setDocumentsOpen((open) => !open)}
              title="Documents and updates"
              aria-label="Documents and updates"
              aria-expanded={documentsOpen}
            >
              Documents
            </button>
          </div>
          <div className="header-center">
            <h1>Radiation Safety RAG</h1>
            <p>Query IAEA and Danish legislation documents</p>
          </div>
          <div className="header-right">
            <ModelSelector value={model} onChange={setModel} />
            <button
              type="button"
              className="header-btn"
              onClick={() => setSettingsOpen(true)}
              title="Settings"
              aria-label="Settings"
            >
              ⚙ Settings
            </button>
          </div>
        </header>
        <SettingsModal isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />
      <div className="conversation-area">
        {messages.length === 0 && serverHasLlmKey === false && !hasAnyApiKeyInStorage() && (
          <div className="api-keys-hint" role="status">
            <h3>API key needed</h3>
            <p>
              To send queries, provide at least one LLM API key. You can use:
            </p>
            <ul>
              <li><strong>Mistral</strong> — <code>MISTRAL_API_KEY</code> in the server&apos;s <code>.env</code>, or add in Settings</li>
              <li><strong>Gemini (Google)</strong> — <code>GOOGLE_API_KEY</code> in <code>.env</code>, or add in Settings</li>
              <li><strong>OpenAI</strong> — <code>OPENAI_API_KEY</code> in <code>.env</code>, or add in Settings</li>
            </ul>
            <p className="api-keys-hint-storage">
              Keys are stored only in your browser and are cleared when you close the tab or leave the page.
            </p>
            <button
              type="button"
              className="api-keys-hint-btn"
              onClick={() => setSettingsOpen(true)}
            >
              Open Settings
            </button>
          </div>
        )}
        <ResponseDisplay messages={messages} />
      </div>
      <div className="input-area">
        {error && (() => {
          const errorLower = error.toLowerCase()
          const isRateLimit = errorLower.includes('rate limit')
          return (
            <p className={isRateLimit ? 'error error-rate-limit' : 'error'}>
              {error}
              {isRateLimit && (
                <span className="error-hint"> You can switch to another model in Settings.</span>
              )}
            </p>
          )
        })()}
        <QueryForm onSubmit={handleSubmit} loading={loading} disabled={false} />
        {messages.some((m) => m.role === 'assistant') && (
          <div className="followup-suggestions">
            <span className="followup-label">Follow-up:</span>
            <button
              type="button"
              className="followup-chip"
              onClick={() => handleSubmit('Can you explain that in more detail?')}
              disabled={loading}
            >
              Can you explain that in more detail?
            </button>
            <button
              type="button"
              className="followup-chip"
              onClick={() => handleSubmit('What sources and limits apply to this?')}
              disabled={loading}
            >
              What sources and limits apply to this?
            </button>
          </div>
        )}
      </div>
      </div>
    </div>
  )
}
