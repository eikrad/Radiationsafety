import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import type { Message } from '../types'
import { QueryProgress } from './QueryProgress'

interface ResponseDisplayProps {
  messages: Message[]
  /** A question sent but not yet answered; shown at once, with progress below it. */
  pendingQuestion?: string | null
}

export function ResponseDisplay({ messages, pendingQuestion = null }: ResponseDisplayProps) {
  const end = useRef<HTMLDivElement>(null)

  // Follow the conversation: bring each new question and answer into view.
  useEffect(() => {
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    end.current?.scrollIntoView?.({ block: 'end', behavior: reduceMotion ? 'auto' : 'smooth' })
  }, [messages.length, pendingQuestion])

  if (messages.length === 0 && !pendingQuestion) return null

  return (
    <div className="response-display">
      {messages.map((msg, i) => (
        <div key={i} className={`message message-${msg.role}`}>
          <h3>{msg.role === 'user' ? 'You' : 'AI Assistant'}</h3>
          {msg.role === 'assistant' && msg.warning && (
            <p className="message-warning">{msg.warning}</p>
          )}
          {msg.role === 'assistant' && msg.used_web_search && (
            <p className="message-meta" title="Brave Search was used this turn">
              {msg.used_web_search_label ?? 'Sources incl. web search'}
            </p>
          )}
          {msg.role === 'assistant' ? (
            <div className="message-text message-text--markdown">
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          ) : (
            <p className="message-text">{msg.content}</p>
          )}
          {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
            <div className="sources">
              <h4>Sources</h4>
              <ul>
                {msg.sources.map((s, j) => (
                  <li key={j}>
                    {s.source.startsWith('http://') || s.source.startsWith('https://') ? (
                      <a className="source source-link" href={s.source} target="_blank" rel="noopener noreferrer">
                        {s.source}
                      </a>
                    ) : (
                      <span className="source">{s.source}</span>
                    )}
                    {s.document_type && (
                      <span className="doc-type"> ({s.document_type})</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ))}
      {pendingQuestion && (
        <>
          <div className="message message-user">
            <h3>You</h3>
            <p className="message-text">{pendingQuestion}</p>
          </div>
          <QueryProgress />
        </>
      )}
      <div ref={end} />
    </div>
  )
}
