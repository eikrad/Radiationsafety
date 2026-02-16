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
  has_local_file?: boolean
  error: string | null
}

interface DocumentsPanelProps {
  onClose: () => void
}

export function DocumentsPanel({ onClose }: DocumentsPanelProps) {
  const [sources, setSources] = useState<DocumentSourceStatus[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [ingesting, setIngesting] = useState(false)
  const [ingestMessage, setIngestMessage] = useState<string | null>(null)
  const [building, setBuilding] = useState(false)
  const [buildMessage, setBuildMessage] = useState<string | null>(null)
  const [addingPdf, setAddingPdf] = useState(false)
  const [addPdfMessage, setAddPdfMessage] = useState<string | null>(null)
  const [editingUrlSourceId, setEditingUrlSourceId] = useState<string | null>(null)
  const [editUrlValue, setEditUrlValue] = useState('')
  const [editUrlSaving, setEditUrlSaving] = useState(false)
  const [editUrlError, setEditUrlError] = useState<string | null>(null)
  const [lookupUrlSourceId, setLookupUrlSourceId] = useState<string | null>(null)
  const [lookupUrlError, setLookupUrlError] = useState<string | null>(null)
  const [downloadUpdateSourceId, setDownloadUpdateSourceId] = useState<string | null>(null)
  const [downloadUpdateError, setDownloadUpdateError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const ingestPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchCheckUpdates = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/documents/check-updates`)
      const data = (await res.json()) as { sources?: DocumentSourceStatus[]; detail?: string }
      if (!res.ok) throw new Error(data.detail ?? 'Failed to check updates')
      setSources(data.sources ?? [])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to check updates')
      setSources([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchCheckUpdates()
  }, [fetchCheckUpdates])

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

  async function buildFromLocal() {
    setBuilding(true)
    setBuildMessage(null)
    try {
      const res = await fetch(`${API_BASE}/documents/build-from-local`, { method: 'POST' })
      const data = (await res.json()) as { sources?: DocumentSourceStatus[]; message?: string; detail?: string }
      if (!res.ok) throw new Error(data.detail ?? data.message ?? 'Build failed')
      setBuildMessage(data.message ?? 'Document list updated.')
      if (data.sources?.length) setSources(data.sources)
      void fetchCheckUpdates()
    } catch (e) {
      setBuildMessage(e instanceof Error ? e.message : 'Build failed')
    } finally {
      setBuilding(false)
    }
  }

  function openAddPdfFilePicker() {
    setAddPdfMessage(null)
    fileInputRef.current?.click()
  }

  async function onAddPdfFileSelected(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setAddPdfMessage('Please select a PDF file.')
      return
    }
    setAddingPdf(true)
    setAddPdfMessage(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('folder', 'IAEA_other')
      const res = await fetch(`${API_BASE}/documents/add-pdf`, {
        method: 'POST',
        body: form,
      })
      const data = (await res.json()) as {
        message?: string
        chunks_added?: number
        url_found?: boolean
        url?: string | null
        detail?: string
      }
      if (!res.ok) throw new Error(data.detail ?? 'Failed to add PDF')
      setAddPdfMessage(data.message ?? 'PDF added.')
      void fetchCheckUpdates()
    } catch (e) {
      setAddPdfMessage(e instanceof Error ? e.message : 'Failed to add PDF')
    } finally {
      setAddingPdf(false)
    }
  }

  const hasOutdated = sources.some((s) => s.update_available)

  function startEditUrl(source: DocumentSourceStatus) {
    setEditUrlError(null)
    setEditingUrlSourceId(source.id)
    setEditUrlValue(source.url || source.download_url || '')
  }

  function cancelEditUrl() {
    setEditingUrlSourceId(null)
    setEditUrlValue('')
    setEditUrlError(null)
  }

  async function runLookupUrl(source: DocumentSourceStatus) {
    setLookupUrlError(null)
    setDownloadUpdateError(null)
    setLookupUrlSourceId(source.id)
    try {
      const res = await fetch(
        `${API_BASE}/documents/source/${encodeURIComponent(source.id)}/lookup-url`,
        { method: 'POST' },
      )
      const data = (await res.json()) as { url?: string; updated?: boolean; detail?: string }
      if (!res.ok) throw new Error(data.detail ?? 'URL not found')
      void fetchCheckUpdates()
    } catch (e) {
      setLookupUrlError(e instanceof Error ? e.message : 'Could not find URL')
    } finally {
      setLookupUrlSourceId(null)
    }
  }

  async function runDownloadUpdate(source: DocumentSourceStatus) {
    setDownloadUpdateError(null)
    setLookupUrlError(null)
    setDownloadUpdateSourceId(source.id)
    try {
      const res = await fetch(
        `${API_BASE}/documents/source/${encodeURIComponent(source.id)}/download-update`,
        { method: 'POST' },
      )
      const data = (await res.json()) as { message?: string; detail?: string }
      if (!res.ok) throw new Error(data.detail ?? 'Download failed')
      void fetchCheckUpdates()
    } catch (e) {
      setDownloadUpdateError(e instanceof Error ? e.message : 'Download failed')
    } finally {
      setDownloadUpdateSourceId(null)
    }
  }

  async function saveSourceUrl() {
    if (!editingUrlSourceId) return
    const url = editUrlValue.trim()
    if (!url) {
      setEditUrlError('Enter a URL')
      return
    }
    setEditUrlSaving(true)
    setEditUrlError(null)
    try {
      const res = await fetch(`${API_BASE}/documents/source/${encodeURIComponent(editingUrlSourceId)}/url`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      })
      const data = (await res.json()) as { detail?: string; message?: string }
      if (!res.ok) throw new Error(data.detail ?? data.message ?? 'Failed to update URL')
      cancelEditUrl()
      void fetchCheckUpdates()
    } catch (e) {
      setEditUrlError(e instanceof Error ? e.message : 'Failed to update URL')
    } finally {
      setEditUrlSaving(false)
    }
  }

  return (
    <aside className="documents-panel" aria-label="Documents list">
      <div className="documents-panel-header">
        <h2 className="documents-panel-title">Documents</h2>
        <button
          type="button"
          className="documents-panel-close"
          onClick={onClose}
          aria-label="Close documents list"
        >
          ×
        </button>
      </div>
      <div className="documents-panel-content">
        {loading && sources.length === 0 && (
          <p className="documents-panel-loading">Checking documents…</p>
        )}
        {error && <p className="documents-panel-error">{error}</p>}
        {!loading && sources.length === 0 && !error && (
          <p className="documents-panel-empty">No document sources configured.</p>
        )}
        <ul className="documents-panel-list">
          {sources.map((s) => {
            const hasLocalFile = s.has_local_file === true
            const openUrl = s.download_url || s.url
            const canOpenLocal = hasLocalFile
            const canOpenWeb = !!openUrl
            const canOpen = canOpenLocal || canOpenWeb
            const hasUrl = !!(s.url || s.download_url)
            const isEditing = editingUrlSourceId === s.id
            return (
            <li key={s.id} className="documents-panel-item">
              <div className="documents-panel-item-row">
                <button
                  type="button"
                  className={`documents-panel-item-btn ${s.update_available ? 'documents-panel-item-btn--outdated' : 'documents-panel-item-btn--current'} ${canOpen ? 'documents-panel-item-btn--clickable' : ''}`}
                  onClick={() => {
                    if (canOpenLocal) {
                      window.open(`${API_BASE}/documents/source/${encodeURIComponent(s.id)}/file`, '_blank', 'noopener,noreferrer')
                    } else if (canOpenWeb) {
                      window.open(openUrl, '_blank', 'noopener,noreferrer')
                    }
                  }}
                  title={
                    canOpen
                      ? canOpenLocal
                        ? 'Open local PDF'
                        : s.update_available
                          ? 'Update available – open web source'
                          : 'Open document (web)'
                      : 'Local only (no URL)'
                  }
                >
                  <span
                    className={`documents-panel-dot ${s.update_available ? 'documents-panel-dot--outdated' : 'documents-panel-dot--current'}`}
                    aria-hidden
                  />
                  <span className="documents-panel-item-body">
                    <span className="documents-panel-item-name">{s.name}</span>
                    <span className="documents-panel-item-version">
                      {s.current_version ? `Current: ${s.current_version}` : 'Not yet ingested'}
                    </span>
                    {s.remote_version && s.remote_version !== s.current_version && (
                      <span className="documents-panel-item-remote">
                        {s.update_available ? 'Newer: ' : 'Remote: '}
                        {s.remote_version}
                      </span>
                    )}
                  </span>
                </button>
                <div className="documents-panel-item-actions">
                  {s.update_available && (
                    <button
                      type="button"
                      className="documents-panel-item-download-update-btn"
                      onClick={(e) => {
                        e.stopPropagation()
                        runDownloadUpdate(s)
                      }}
                      disabled={downloadUpdateSourceId === s.id}
                      title="Download new version and backup the current one"
                    >
                      {downloadUpdateSourceId === s.id ? 'Downloading…' : 'Download update'}
                    </button>
                  )}
                  <button
                    type="button"
                    className="documents-panel-item-edit-url-link"
                    onClick={(e) => {
                      e.stopPropagation()
                      startEditUrl(s)
                    }}
                    onContextMenu={(e) => {
                      e.preventDefault()
                      startEditUrl(s)
                    }}
                    title={hasUrl ? 'Edit URL (or right-click)' : 'Add URL (or right-click)'}
                  >
                    {hasUrl ? 'Edit' : 'Add URL'}
                  </button>
                  <button
                    type="button"
                    className="documents-panel-item-search-url-link"
                    onClick={(e) => {
                      e.stopPropagation()
                      runLookupUrl(s)
                    }}
                    disabled={lookupUrlSourceId === s.id}
                    title={(s.folder || '') === 'Bekendtgørelse' ? 'Find URL from retsinformation.dk (API or page)' : 'Find URL from iaea.org search'}
                  >
                    {lookupUrlSourceId === s.id ? 'Searching…' : 'Search URL'}
                  </button>
                </div>
              </div>
              {isEditing && (
                <div className="documents-panel-item-url-form">
                  <input
                    type="url"
                    className="documents-panel-item-url-input"
                    value={editUrlValue}
                    onChange={(e) => setEditUrlValue(e.target.value)}
                    placeholder="https://www.iaea.org/... or retsinformation.dk/..."
                    aria-label="Document URL"
                  />
                  <div className="documents-panel-item-url-actions">
                    <button
                      type="button"
                      className="documents-panel-item-url-save"
                      onClick={saveSourceUrl}
                      disabled={editUrlSaving}
                    >
                      {editUrlSaving ? 'Saving…' : 'Save'}
                    </button>
                    <button
                      type="button"
                      className="documents-panel-item-url-cancel"
                      onClick={cancelEditUrl}
                      disabled={editUrlSaving}
                    >
                      Cancel
                    </button>
                  </div>
                  {editUrlError && (
                    <p className="documents-panel-item-url-error">{editUrlError}</p>
                  )}
                </div>
              )}
            </li>
          )})}
        </ul>
        {(lookupUrlError || downloadUpdateError) && (
          <p className="documents-panel-lookup-error" role="alert">
            {downloadUpdateError || lookupUrlError}
          </p>
        )}
        {hasOutdated && (
          <p className="documents-panel-hint">
            Use “Download update” to fetch the new version and backup the current one. Then run ingestion below.
          </p>
        )}
        {!hasOutdated && sources.some((s) => s.has_local_file || s.download_url || s.url) && (
          <p className="documents-panel-hint">
            Click a document to open the local PDF, or the web page if no local file.
          </p>
        )}
        <div className="documents-panel-build">
          <button
            type="button"
            className="documents-panel-build-btn"
            onClick={buildFromLocal}
            disabled={building}
            title="Scan documents/ folders, extract versions from PDFs, and update document_sources.yaml"
          >
            {building ? 'Building…' : 'Build list from local PDFs'}
          </button>
          {buildMessage && (
            <p className="documents-panel-build-msg">{buildMessage}</p>
          )}
        </div>
        <div className="documents-panel-add-pdf">
          <p className="documents-panel-add-pdf-hint">
            Only PDFs from retsinformation.dk and the IAEA website can be added. The document is added to the collection and its URL is looked up from the extracted title.
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,application/pdf"
            className="documents-panel-add-pdf-input"
            aria-hidden
            onChange={onAddPdfFileSelected}
          />
          <button
            type="button"
            className="documents-panel-add-pdf-btn"
            onClick={openAddPdfFilePicker}
            disabled={addingPdf}
            title="Add a PDF to the collection and look up its publication URL (IAEA or retsinformation.dk)"
          >
            {addingPdf ? 'Adding…' : 'Add PDF from computer'}
          </button>
          {addPdfMessage && (
            <p className="documents-panel-add-pdf-msg">{addPdfMessage}</p>
          )}
        </div>
        <div className="documents-panel-ingest">
          <button
            type="button"
            className="documents-panel-ingest-btn"
            onClick={runIngest}
            disabled={ingesting}
          >
            {ingesting ? 'Ingestion running…' : 'Start ingestion'}
          </button>
          {ingesting && (
            <div className="documents-panel-ingest-progress" role="progressbar" aria-valuetext="Ingestion in progress">
              <div className="documents-panel-ingest-progress-bar" />
            </div>
          )}
          {ingestMessage && (
            <p className="documents-panel-ingest-msg">{ingestMessage}</p>
          )}
        </div>
      </div>
    </aside>
  )
}
