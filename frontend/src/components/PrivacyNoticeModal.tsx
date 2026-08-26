import ReactMarkdown from 'react-markdown'
import privacyNoticeContent from '../content/privacy-notice.md?raw'

interface PrivacyNoticeModalProps {
  isOpen: boolean
  onClose: () => void
}

export function PrivacyNoticeModal({ isOpen, onClose }: PrivacyNoticeModalProps) {
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
          <ReactMarkdown>{privacyNoticeContent}</ReactMarkdown>
        </div>
      </div>
    </div>
  )
}
