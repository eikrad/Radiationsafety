import ReactMarkdown from 'react-markdown'
import privacyNoticeContent from '../content/privacy-notice.md?raw'

interface PrivacyNoticeModalProps {
  isOpen: boolean
  onClose: () => void
  /** From GET /api/config. Both null when the operator hasn't set PRIVACY_CONTROLLER_NAME/CONTACT. */
  controllerName: string | null
  controllerContact: string | null
}

export function PrivacyNoticeModal({
  isOpen,
  onClose,
  controllerName,
  controllerContact,
}: PrivacyNoticeModalProps) {
  if (!isOpen) return null

  return (
    <div className="settings-overlay" onClick={onClose}>
      <div
        className="settings-modal privacy-notice-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="settings-header">
          <h2>Privacy notice</h2>
          <button type="button" className="settings-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <div className="privacy-notice-body">
          <h3>Who operates this instance</h3>
          {controllerName ? (
            <p>
              <strong>Controller:</strong> {controllerName}
              {controllerContact && (
                <>
                  {' — '}
                  <strong>Contact:</strong> {controllerContact}
                </>
              )}
            </p>
          ) : (
            <p>
              This deployment hasn't been configured with a separate operator identity
              (<code>PRIVACY_CONTROLLER_NAME</code>). That usually means it's running
              locally for your own use — in which case you are the only party processing
              your own data, and there is no separate operator to name. If someone else
              runs this instance for you, ask them to set{' '}
              <code>PRIVACY_CONTROLLER_NAME</code> and <code>PRIVACY_CONTROLLER_CONTACT</code>{' '}
              so this notice can identify them.
            </p>
          )}
          <ReactMarkdown>{privacyNoticeContent}</ReactMarkdown>
        </div>
      </div>
    </div>
  )
}
