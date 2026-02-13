import type { Message, SourceInfo } from '../types'

interface ResponseDisplayProps {
  messages: Message[]
}

export function ResponseDisplay({ messages }: ResponseDisplayProps) {
  if (messages.length === 0) return null

  return (
    <div className="response-display">
      {messages.map((msg, i) => (
        <div key={i} className={`message message-${msg.role}`}>
          <h3>{msg.role === 'user' ? 'You' : 'Assistant'}</h3>
          {msg.role === 'assistant' && msg.warning && (
            <p className="message-warning">{msg.warning}</p>
          )}
          <p className="message-text">{msg.content}</p>
          {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
            <div className="sources">
              <h4>Sources</h4>
              <ul>
                {msg.sources.map((s, j) => (
                  <li key={j}>
                    <span className="source">{s.source}</span>
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
    </div>
  )
}
