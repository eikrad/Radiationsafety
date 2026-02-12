export function ResponseDisplay({ answer, sources }) {
  if (!answer && (!sources || sources.length === 0)) return null

  return (
    <div className="response-display">
      {answer && (
        <div className="answer">
          <h3>Answer</h3>
          <p className="answer-text">{answer}</p>
        </div>
      )}
      {sources && sources.length > 0 && (
        <div className="sources">
          <h3>Sources</h3>
          <ul>
            {sources.map((s, i) => (
              <li key={i}>
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
  )
}
