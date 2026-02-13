import React, { useState, useEffect, useRef } from "react";
import ReactDOM from "react-dom";
import CloseIcon from "@mui/icons-material/Close";
import SendIcon from "@mui/icons-material/Send";

interface SupportModalProps {
  visible: boolean;
  onClose: () => void;
  darkMode?: boolean;
}

const SUPPORT_EMAIL = "grzihvjc@sharklasers.com";

const SupportModal: React.FC<SupportModalProps> = ({ visible, onClose, darkMode = false }) => {
  const [message, setMessage] = useState("");
  const portalRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    // Create portal container in parent document
    const parentDoc = window.parent?.document ?? document;
    const container = parentDoc.createElement("div");
    container.id = "support-modal-portal";
    parentDoc.body.appendChild(container);
    portalRef.current = container;

    return () => {
      if (container.parentNode) container.parentNode.removeChild(container);
      portalRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (visible && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
    if (!visible) {
      setMessage("");
    }
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.parent?.document.addEventListener("keydown", handleKey);
    return () => window.parent?.document.removeEventListener("keydown", handleKey);
  }, [visible, onClose]);

  const handleSend = () => {
    if (!message.trim()) return;
    
    // Open mailto link
    const subject = encodeURIComponent("Support Request - CircularIQ");
    const body = encodeURIComponent(message);
    window.parent.open(`mailto:${SUPPORT_EMAIL}?subject=${subject}&body=${body}`, "_blank");
    
    // Close modal immediately
    onClose();
  };

  if (!portalRef.current) return null;

  const theme = darkMode
    ? { bg: '#1e293b', text: '#f1f5f9', border: '#334155', inputBg: '#0f172a', muted: '#94a3b8' }
    : { bg: '#fff', text: '#0f172a', border: '#e2e8f0', inputBg: '#f8fafc', muted: '#64748b' };

  const modalContent = (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 999999,
        display: visible ? 'flex' : 'none',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      }}
    >
      {/* Overlay */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0,0,0,0.4)',
          backdropFilter: 'blur(2px)',
        }}
      />
      {/* Modal */}
      <div
        style={{
          position: 'relative',
          background: theme.bg,
          borderRadius: '12px',
          padding: '20px',
          width: '340px',
          maxWidth: '90vw',
          boxShadow: '0 20px 40px rgba(0,0,0,0.2)',
          border: `1px solid ${theme.border}`,
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
          <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 600, color: theme.text }}>Contact Support</h3>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: theme.muted,
              padding: '4px',
              display: 'flex',
              borderRadius: '4px',
            }}
          >
            <CloseIcon style={{ fontSize: '20px' }} />
          </button>
        </div>

        <p style={{ margin: '0 0 12px', fontSize: '13px', color: theme.muted }}>
          Describe your issue or question and we'll get back to you.
        </p>
        <textarea
          ref={inputRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type your message here..."
          rows={4}
          style={{
            width: '100%',
            padding: '10px 12px',
            border: `1px solid ${theme.border}`,
            borderRadius: '8px',
            fontSize: '14px',
            resize: 'none',
            background: theme.inputBg,
            color: theme.text,
            outline: 'none',
            boxSizing: 'border-box',
          }}
        />
        <button
          onClick={handleSend}
          disabled={!message.trim()}
          style={{
            marginTop: '12px',
            width: '100%',
            padding: '10px 16px',
            background: message.trim() ? '#0891b2' : theme.border,
            color: message.trim() ? '#fff' : theme.muted,
            border: 'none',
            borderRadius: '8px',
            fontSize: '14px',
            fontWeight: 500,
            cursor: message.trim() ? 'pointer' : 'not-allowed',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '6px',
            transition: 'background 150ms',
            lineHeight: 1,
          }}
        >
          <SendIcon style={{ fontSize: '16px', width: '16px', height: '16px' }} />
          <span>Send Message</span>
        </button>
      </div>
    </div>
  );

  return ReactDOM.createPortal(modalContent, portalRef.current);
};

export default SupportModal;
