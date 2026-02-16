import { useCallback, useEffect, useRef, useState } from 'react'

const API_BASE = '/api'

export interface DocumentSourceStatus {
  id: string
  name: string
  url: string
  folder: string
  current_version: string | null
  remote_version: string | null
  update_available: boolean
  local_date: string | null
  remote_date: string | null
  download_url: string
  error: string | null
}

interface CheckUpdatesResponse {
  sources: DocumentSourceStatus[]
  recent_iaea: { title: string; link: string; published: string }[]
}

interface DocumentUpdatesModalProps {
  isOpen: boolean
  onClose: () => void
  onSourcesChange?: (sources: DocumentSourceStatus[]) => void
}

export function DocumentUpdatesModal({ isOpen, onClose, onSourcesChange }: DocumentUpdatesModalProps) {
  const [sources, setSources] = useState<DocumentSourceStatus[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [ingesting, setIngesting] = useState(false)
  const [ingestMessage, setIngestMessage] = useState<string | null>(null)
  const ingestPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchCheckUpdates = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/documents/check-updates`)
      const data = (await res.json()) as CheckUpdatesResponse & { detail?: string }
      if (!res.ok) throw new Error(data.detail ?? 'Failed to check updates')
      const next = data.sources ?? []
      setSources(next)
      onSourcesChange?.(next)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to check updates')
      setSources([])
      onSourcesChange?.([])
    } finally {
      setLoading(false)
    }
  }, [onSourcesChange])

  useEffect(() => {
    if (isOpen) fetchCheckUpdates()
  }, [isOpen, fetchCheckUpdates])

  useEffect(() => {
    return () => {
      if (ingestPollRef.current) window.clearInterval(ingestPollRef.current)
    }
  }, [])

  async function runIngest() {
    setIngesting(true)
    setIngestMessage('Starting ingestion…')
    try {
      const res = await fetch(`${API_BASE}/ingest`, { method: 'POST' })
      const data = (await res.json()) as { status?: string; detail?: string; message?: string }
      if (!res.ok) throw new Error(data.detail ?? data.message ?? 'Failed to start ingestion')
      setIngestMessage('Ingestion started. This may take several minutes.')
      ingestPollRef.current = window.setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE}/ingest/status`)
          const statusData = (await statusRes.json()) as { status?: string }
          if (statusData.status === 'idle') {
            if (ingestPollRef.current) window.clearInterval(ingestPollRef.current)
            ingestPollRef.current = null
            setIngesting(false)
            setIngestMessage('Ingestion finished.')
            void fetchCheckUpdates()
          }
        } catch {
          if (ingestPollRef.current) window.clearInterval(ingestPollRef.current)
          ingestPollRef.current = null
          setIngesting(false)
        }
      }, 3000)
    } catch (e) {
      setIngestMessage(e instanceof Error ? e.message : 'Failed to start ingestion')
      setIngesting(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div className="settings-modal documents-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>Documents &amp; updates</h2>
          <button type="button" className="settings-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <p className="settings-hint">
          Registered sources are listed below. Use “Check for updates” to see if newer versions are
          available; open the link to download, then run “Re-run ingestion” after updating files.
        </p>
        <div className="documents-actions">
          <button
            type="button"
            className="settings-save"
            onClick={fetchCheckUpdates}
            disabled={loading}
          >
            {loading ? 'Checking…' : 'Check for updates'}
          </button>
        </div>
        {error && <p className="documents-error">{error}</p>}
        <div className="documents-list">
          {sources.length === 0 && !loading && !error && (
            <p className="documents-empty">No document sources configured. Add document_sources.yaml (see document_sources.example.yaml).</p>
          )}
          {sources.map((s) => (
            <div key={s.id} className="documents-item">
              <div className="documents-item-name">{s.name}</div>
              <div className="documents-item-versions">
                <span className="documents-version">
                  Current: {s.current_version ?? '—'}
                </span>
                {s.remote_version &&
                  !(
                    s.update_available &&
                    !s.current_version
                  ) && (
                    <span className="documents-version">
                      Remote: {s.remote_version}
                    </span>
                  )}
              </div>
              {s.update_available && s.current_version && s.remote_version && (
                <p className="documents-update-msg">
                  Update available: from <strong>{s.current_version}</strong> to{' '}
                  <strong>{s.remote_version}</strong>
                </p>
              )}
              {s.update_available && !s.current_version && s.remote_version && (
                <p className="documents-update-msg">
                  {s.remote_version.startsWith('Cannot detect') ? (
                    <span>{s.remote_version}</span>
                  ) : (
                    <>
                      This entry points to an older publication. Current
                      edition: <strong>{s.remote_version}</strong>
                    </>
                  )}
                </p>
              )}
              {s.error && <p className="documents-error-inline">{s.error}</p>}
              <a
                href={s.download_url}
                target="_blank"
                rel="noopener noreferrer"
                className="documents-link"
              >
                Open source / Download
              </a>
            </div>
          ))}
        </div>
        <div className="documents-ingest">
          <button
            type="button"
            className="settings-save"
            onClick={runIngest}
            disabled={ingesting}
          >
            {ingesting ? 'Ingestion running…' : 'Re-run ingestion'}
          </button>
          {ingestMessage && <p className="documents-ingest-msg">{ingestMessage}</p>}
        </div>
        <div className="settings-actions">
          <button type="button" className="settings-cancel" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
