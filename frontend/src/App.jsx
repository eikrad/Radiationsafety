import { useState } from 'react'
import { QueryForm } from './components/QueryForm'
import { ResponseDisplay } from './components/ResponseDisplay'
import './App.css'

const API_BASE = '/api'

export default function App() {
  const [loading, setLoading] = useState(false)
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [error, setError] = useState('')

  async function handleSubmit(question) {
    setLoading(true)
    setError('')
    setAnswer('')
    setSources([])
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Request failed')
      setAnswer(data.answer || '')
      setSources(data.sources || [])
    } catch (err) {
      setError(err.message || 'Failed to get answer. Is the backend running?')
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
      <QueryForm onSubmit={handleSubmit} loading={loading} disabled={false} />
      {error && <p className="error">{error}</p>}
      <ResponseDisplay answer={answer} sources={sources} />
    </div>
  )
}
