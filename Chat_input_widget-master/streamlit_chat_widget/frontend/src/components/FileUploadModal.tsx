import React, { useState, useCallback, useRef, useEffect } from "react";
import ReactDOM from "react-dom";

interface FileUploadModalProps {
  visible: boolean;
  onClose: () => void;
  onFileSelect?: (file: File) => void;
  darkMode?: boolean;
}

// Color themes
const lightTheme = {
  bg: '#fff',
  border: '#f1f5f9',
  text: '#0f172a',
  textMuted: '#64748b',
  textLabel: '#334155',
  closeBg: '#f1f5f9',
  closeHover: '#e2e8f0',
  dropBorder: '#cbd5e1',
  dropBg: '#fafbfc',
  dropHoverBg: '#f0fdfa',
  dropHoverBorder: '#0891b2',
  dropDragBg: '#ecfeff',
  dropSuccessBg: '#ecfdf5',
  dropSuccessBorder: '#10b981',
  selectedBg: '#f0fdf4',
  accent: '#0891b2',
  accentHover: '#0e7490',
  btnCancelBg: '#f1f5f9',
  btnCancelText: '#475569',
  btnDisabled: '#cbd5e1',
  overlay: 'rgba(15,23,42,0.5)',
  shadow: 'rgba(15,23,42,0.2)',
  removeBg: '#fef2f2',
  removeHover: '#fee2e2',
  removeText: '#ef4444',
};

const darkTheme = {
  bg: '#1e293b',
  border: '#334155',
  text: '#f1f5f9',
  textMuted: '#94a3b8',
  textLabel: '#e2e8f0',
  closeBg: '#334155',
  closeHover: '#475569',
  dropBorder: '#475569',
  dropBg: '#0f172a',
  dropHoverBg: '#0f172a',
  dropHoverBorder: '#14b8a6',
  dropDragBg: '#134e4a',
  dropSuccessBg: '#14532d',
  dropSuccessBorder: '#22c55e',
  selectedBg: '#14532d',
  accent: '#14b8a6',
  accentHover: '#0d9488',
  btnCancelBg: '#334155',
  btnCancelText: '#e2e8f0',
  btnDisabled: '#475569',
  overlay: 'rgba(0,0,0,0.6)',
  shadow: 'rgba(0,0,0,0.4)',
  removeBg: '#450a0a',
  removeHover: '#7f1d1d',
  removeText: '#fca5a5',
};

const FileUploadModal: React.FC<FileUploadModalProps> = ({ visible, onClose, onFileSelect, darkMode = false }) => {
  const portalRef = useRef<HTMLDivElement | null>(null);
  const [targetDoc, setTargetDoc] = useState<Document | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [portalReady, setPortalReady] = useState(false);

  const theme = darkMode ? darkTheme : lightTheme;

  // Create portal once on mount - always keep it ready
  useEffect(() => {
    let doc: Document = document;
    try {
      if (window.parent && window.parent.document && window.parent.document.body) {
        doc = window.parent.document;
      }
    } catch (e) {}
    setTargetDoc(doc);
    
    const portal = doc.createElement("div");
    portal.id = "file-upload-modal-portal";
    doc.body.appendChild(portal);
    portalRef.current = portal;
    setPortalReady(true);
    
    return () => {
      if (portal.parentNode) {
        portal.parentNode.removeChild(portal);
      }
      portalRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!visible) {
      setSelectedFile(null);
      setIsDragging(false);
    }
  }, [visible]);

  useEffect(() => {
    if (!visible || !targetDoc) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    targetDoc.addEventListener("keydown", onKeyDown);
    return () => targetDoc.removeEventListener("keydown", onKeyDown);
  }, [visible, onClose, targetDoc]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      setSelectedFile(files[0]);
    }
  }, []);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setSelectedFile(files[0]);
    }
  }, []);

  const handleUpload = useCallback(() => {
    if (selectedFile && onFileSelect) {
      onFileSelect(selectedFile);
    }
    onClose();
  }, [selectedFile, onFileSelect, onClose]);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  // Don't render anything until portal is ready
  if (!portalReady || !portalRef.current) return null;
  
  // When not visible, render empty portal (no overlay, no modal content)
  if (!visible) {
    return ReactDOM.createPortal(null, portalRef.current);
  }

  // Styles
  const wrapperStyle: React.CSSProperties = {
    position: 'fixed',
    inset: 0,
    zIndex: 999999,
    pointerEvents: visible ? 'auto' : 'none',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  const overlayStyle: React.CSSProperties = {
    position: 'fixed',
    inset: 0,
    background: theme.overlay,
    opacity: visible ? 1 : 0,
    transition: 'opacity 200ms ease',
    pointerEvents: visible ? 'auto' : 'none',
    backdropFilter: 'blur(4px)',
  };

  const modalStyle: React.CSSProperties = {
    background: theme.bg,
    borderRadius: '16px',
    boxShadow: `0 20px 60px ${theme.shadow}`,
    width: '440px',
    maxWidth: '92vw',
    transform: visible ? 'scale(1) translateY(0)' : 'scale(0.95) translateY(10px)',
    opacity: visible ? 1 : 0,
    transition: 'all 200ms cubic-bezier(.4,0,.2,1)',
    pointerEvents: 'auto',
    overflow: 'hidden',
  };

  const headerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '18px 24px',
    borderBottom: `1px solid ${theme.border}`,
  };

  const titleStyle: React.CSSProperties = {
    margin: 0,
    fontSize: '17px',
    fontWeight: 600,
    color: theme.text,
  };

  const closeButtonStyle: React.CSSProperties = {
    border: 'none',
    background: theme.closeBg,
    color: theme.textMuted,
    cursor: 'pointer',
    fontSize: '16px',
    width: '32px',
    height: '32px',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 150ms',
  };

  const bodyStyle: React.CSSProperties = {
    padding: '24px',
  };

  const getDropZoneStyle = (): React.CSSProperties => {
    let borderColor = theme.dropBorder;
    let background = theme.dropBg;
    let borderStyle = 'dashed';

    if (selectedFile) {
      borderColor = theme.dropSuccessBorder;
      background = theme.dropSuccessBg;
      borderStyle = 'solid';
    } else if (isDragging) {
      borderColor = theme.accent;
      background = theme.dropDragBg;
      borderStyle = 'solid';
    }

    return {
      border: `2px ${borderStyle} ${borderColor}`,
      borderRadius: '12px',
      padding: '40px 24px',
      textAlign: 'center',
      transition: 'all 200ms',
      cursor: 'pointer',
      background,
    };
  };

  const selectedFileStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '12px 16px',
    background: theme.selectedBg,
    borderRadius: '10px',
    marginTop: '16px',
  };

  const footerStyle: React.CSSProperties = {
    padding: '16px 24px',
    borderTop: `1px solid ${theme.border}`,
    display: 'flex',
    gap: '12px',
    justifyContent: 'flex-end',
  };

  const cancelBtnStyle: React.CSSProperties = {
    border: 'none',
    padding: '10px 20px',
    borderRadius: '10px',
    fontWeight: 600,
    fontSize: '14px',
    cursor: 'pointer',
    background: theme.btnCancelBg,
    color: theme.btnCancelText,
    transition: 'all 150ms',
  };

  const uploadBtnStyle: React.CSSProperties = {
    border: 'none',
    padding: '10px 20px',
    borderRadius: '10px',
    fontWeight: 600,
    fontSize: '14px',
    cursor: selectedFile ? 'pointer' : 'not-allowed',
    background: selectedFile ? theme.accent : theme.btnDisabled,
    color: '#fff',
    transition: 'all 150ms',
  };

  const removeButtonStyle: React.CSSProperties = {
    border: 'none',
    background: theme.removeBg,
    color: theme.removeText,
    width: '28px',
    height: '28px',
    borderRadius: '6px',
    cursor: 'pointer',
    fontSize: '14px',
    transition: 'all 150ms',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  return ReactDOM.createPortal(
    <div style={wrapperStyle}>
      <div style={overlayStyle} onClick={onClose} aria-hidden />
      <div style={modalStyle}>
        <div style={headerStyle}>
          <h3 style={titleStyle}>📎 Attach File</h3>
          <button style={closeButtonStyle} onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div style={bodyStyle}>
          <div
            style={getDropZoneStyle()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div style={{ fontSize: '48px', marginBottom: '12px', opacity: 0.6 }}>📄</div>
            <div style={{ fontSize: '15px', fontWeight: 600, color: theme.textLabel, marginBottom: '6px' }}>
              Drag & drop your file here
            </div>
            <div style={{ fontSize: '13px', color: theme.textMuted }}>
              or click to browse
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
          {selectedFile && (
            <div style={selectedFileStyle}>
              <span style={{ fontSize: '24px' }}>📎</span>
              <div style={{ flex: 1, textAlign: 'left' }}>
                <div style={{ fontSize: '14px', fontWeight: 500, color: theme.text, wordBreak: 'break-all' }}>
                  {selectedFile.name}
                </div>
                <div style={{ fontSize: '12px', color: theme.textMuted }}>
                  {formatFileSize(selectedFile.size)}
                </div>
              </div>
              <button
                style={removeButtonStyle}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedFile(null);
                }}
              >
                ✕
              </button>
            </div>
          )}
        </div>
        <div style={footerStyle}>
          <button style={cancelBtnStyle} onClick={onClose}>Cancel</button>
          <button
            style={uploadBtnStyle}
            onClick={handleUpload}
            disabled={!selectedFile}
          >
            Attach
          </button>
        </div>
      </div>
    </div>,
    portalRef.current
  );
};

export default FileUploadModal;
