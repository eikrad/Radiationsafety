import { useState, FormEvent } from 'react'

interface QueryFormProps {
  onSubmit: (question: string) => void
  loading: boolean
  disabled: boolean
}

export function QueryForm({ onSubmit, loading, disabled }: QueryFormProps) {
  const [question, setQuestion] = useState('')

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    if (question.trim()) {
      onSubmit(question.trim())
    }
  }

  return (
    <form onSubmit={handleSubmit} className="query-form">
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a question about IAEA or Danish radiation safety..."
        rows={3}
        disabled={disabled || loading}
      />
      <button type="submit" disabled={disabled || loading}>
        {loading ? 'Searching...' : 'Ask'}
      </button>
    </form>
  )
}
