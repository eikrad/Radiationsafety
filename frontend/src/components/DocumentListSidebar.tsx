import type { DocumentSourceStatus } from './DocumentUpdatesModal'

interface DocumentListSidebarProps {
  sources: DocumentSourceStatus[]
  loading: boolean
  onOpenModal: () => void
}

export function DocumentListSidebar({ sources, loading, onOpenModal }: DocumentListSidebarProps) {
  return (
    <aside className="documents-sidebar">
      <button
        type="button"
        className="documents-sidebar-btn"
        onClick={onOpenModal}
        title="Documents and updates"
        aria-label="Documents and updates"
      >
        Documents
      </button>
      <ul className="documents-sidebar-list" aria-label="Document sources in use">
        {loading && sources.length === 0 && (
          <li className="documents-sidebar-item documents-sidebar-item--loading">Checking…</li>
        )}
        {!loading && sources.length === 0 && (
          <li className="documents-sidebar-item documents-sidebar-item--empty">No sources</li>
        )}
        {sources.map((s) => (
          <li key={s.id} className="documents-sidebar-item">
            <span
              className={`documents-sidebar-dot ${s.update_available ? 'documents-sidebar-dot--update' : 'documents-sidebar-dot--ok'}`}
              title={s.update_available ? 'Update available' : 'Up to date'}
              aria-hidden
            />
            <span className="documents-sidebar-name" title={s.name}>
              {s.name}
            </span>
          </li>
        ))}
      </ul>
    </aside>
  )
}
