import { useState } from 'react'
import { QueryForm } from './components/QueryForm'
import { ResponseDisplay } from './components/ResponseDisplay'
import type { Message, QueryResponse, SourceInfo } from './types'
import './App.css'

const API_BASE = '/api'

export default function App() {
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [error, setError] = useState('')

  async function handleSubmit(question: string) {
    setLoading(true)
    setError('')
    const chatHistory: [string, string][] = []
    for (let i = 0; i < messages.length - 1; i++) {
      if (messages[i]?.role === 'user' && messages[i + 1]?.role === 'assistant') {
        chatHistory.push([messages[i]!.content, messages[i + 1]!.content])
      }
    }
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, chat_history: chatHistory }),
      })
      const data = (await res.json()) as QueryResponse & { detail?: string }
      if (!res.ok) throw new Error(data.detail ?? 'Request failed')
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
      setError(
        err instanceof Error ? err.message : 'Failed to get answer. Is the backend running?'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Radiation Safety RAG</h1>
        <p>Query IAEA and Danish legislation documents</p>
      </header>
      <div className="conversation-area">
        <ResponseDisplay messages={messages} />
      </div>
      <div className="input-area">
        {error && <p className="error">{error}</p>}
        <QueryForm onSubmit={handleSubmit} loading={loading} disabled={false} />
      </div>
    </div>
  )
}
