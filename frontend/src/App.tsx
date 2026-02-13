import { useState, useEffect } from 'react'
import { ModelSelector } from './components/ModelSelector'
import { QueryForm } from './components/QueryForm'
import { ResponseDisplay } from './components/ResponseDisplay'
import { SettingsModal } from './components/SettingsModal'
import { MODELS, STORAGE_KEYS, type Model } from './constants'
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

function loadApiKeys(): Record<Model, string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.apiKeys)
    if (!raw) return { mistral: '', gemini: '', openai: '' }
    const parsed = JSON.parse(raw) as Record<string, string>
    return {
      mistral: parsed.mistral ?? '',
      gemini: parsed.gemini ?? '',
      openai: parsed.openai ?? '',
    }
  } catch {
    return { mistral: '', gemini: '', openai: '' }
  }
}

function loadModelVariants(): Record<Model, string> {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.modelVariants)
    if (!raw) {
      return { mistral: 'default', gemini: 'gemini-2.5-flash-lite', openai: 'gpt-4o-mini' }
    }
    const parsed = JSON.parse(raw) as Record<string, string>
    return {
      mistral: parsed.mistral ?? 'default',
      gemini: parsed.gemini ?? 'gemini-2.5-flash-lite',
      openai: parsed.openai ?? 'gpt-4o-mini',
    }
  } catch {
    return { mistral: 'default', gemini: 'gemini-2.5-flash-lite', openai: 'gpt-4o-mini' }
  }
}

export default function App() {
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [error, setError] = useState('')
  const [model, setModel] = useState<Model>(loadStoredModel)
  const [settingsOpen, setSettingsOpen] = useState(false)

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
      const lower = msg.toLowerCase()
      if (lower.includes('api key') || lower.includes('rate limit') || lower.includes('quota')) {
        setSettingsOpen(true)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <div className="header-title">
          <h1>Radiation Safety RAG</h1>
          <p>Query IAEA and Danish legislation documents</p>
        </div>
        <div className="header-controls">
          <ModelSelector value={model} onChange={setModel} />
          <button
            type="button"
            className="settings-btn"
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
        {error && (
          <p className={error.toLowerCase().includes('rate limit') ? 'error error-rate-limit' : 'error'}>
            {error}
            {error.toLowerCase().includes('rate limit') && (
              <span className="error-hint"> You can switch to another model in Settings.</span>
            )}
          </p>
        )}
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
  )
}
