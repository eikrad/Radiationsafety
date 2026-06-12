import { useState, FormEvent, KeyboardEvent } from 'react'

interface QueryFormProps {
  onSubmit: (question: string) => void
  loading: boolean
  disabled: boolean
}

export function QueryForm({ onSubmit, loading, disabled }: QueryFormProps) {
  const [question, setQuestion] = useState('')

  function submit() {
    const q = question.trim()
    if (q && !disabled && !loading) {
      onSubmit(q)
      setQuestion('')
    }
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    submit()
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <form onSubmit={handleSubmit} className="query-form">
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={handleKeyDown}
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
