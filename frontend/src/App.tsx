import { useEffect, useState } from 'react'
import { DocumentsPanel } from './components/DocumentsPanel'
import { ModelSelector } from './components/ModelSelector'
import { QueryForm } from './components/QueryForm'
import { ResponseDisplay } from './components/ResponseDisplay'
import { SettingsModal } from './components/SettingsModal'
import { MODELS, STORAGE_KEYS, type Model } from './constants'
import { loadApiKeys, loadModelVariants } from './storage'
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

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEYS.model, model)
    } catch {}
  }, [model])

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
