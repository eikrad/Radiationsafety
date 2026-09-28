import { useEffect, useState } from 'react'

/** After this long, say that a slow answer is still expected to arrive. */
const LONG_WAIT_SEC = 45

/**
 * Stands in for the assistant's answer while /query runs. The backend reports no
 * progress yet, so this shows only that work is going on and for how long; it does
 * not pretend to know which step is running.
 */
export function QueryProgress() {
  const [elapsed, setElapsed] = useState(0)

  useEffect(() => {
    const started = Date.now()
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="message message-assistant message-pending">
      <h3>AI Assistant</h3>
      <div className="query-progress" role="status">
        <span className="query-progress-dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span>Searching the IAEA and Danish documents and writing an answer…</span>
        {/* Hidden from screen readers so the status is not re-announced every second. */}
        <span className="query-progress-elapsed" aria-hidden="true">
          {elapsed} s
        </span>
      </div>
      {elapsed >= LONG_WAIT_SEC && (
        <p className="query-progress-hint">
          Still working – thorough answers can take up to a few minutes.
        </p>
      )}
    </div>
  )
}
