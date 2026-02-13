import React, { useCallback, useRef, useState, useEffect } from "react";
import ReactDOM from "react-dom";

interface FilterState {
  yearStart?: string;
  yearEnd?: string;
  author?: string;
  keywords?: string;
}

interface FilterSidebarProps {
  visible: boolean;
  filters: FilterState;
  onChange: (k: keyof FilterState, v: string) => void;
  onApply: () => void;
  onCancel: () => void;
  darkMode?: boolean;
}

// Color themes
const lightTheme = {
  bg: '#fff',
  bgHeader: '#fafbfc',
  bgFooter: '#fafbfc',
  border: '#e2e8f0',
  borderLight: '#f1f5f9',
  text: '#0f172a',
  textMuted: '#64748b',
  textLabel: '#334155',
  textPlaceholder: '#94a3b8',
  chipBg: '#f1f5f9',
  chipText: '#475569',
  chipBorder: '#e2e8f0',
  chipHover: '#e2e8f0',
  chipSelected: '#0891b2',
  accent: '#0891b2',
  accentHover: '#0e7490',
  closeBg: '#f1f5f9',
  closeHover: '#e2e8f0',
  overlay: 'rgba(15,23,42,0.4)',
  shadow: 'rgba(15,23,42,0.12)',
  inputBg: '#fff',
  btnCancelBg: '#f1f5f9',
  btnCancelText: '#475569',
};

const darkTheme = {
  bg: '#1e293b',
  bgHeader: '#0f172a',
  bgFooter: '#0f172a',
  border: '#334155',
  borderLight: '#334155',
  text: '#f1f5f9',
  textMuted: '#94a3b8',
  textLabel: '#e2e8f0',
  textPlaceholder: '#64748b',
  chipBg: '#334155',
  chipText: '#e2e8f0',
  chipBorder: '#475569',
  chipHover: '#475569',
  chipSelected: '#14b8a6',
  accent: '#14b8a6',
  accentHover: '#0d9488',
  closeBg: '#334155',
  closeHover: '#475569',
  overlay: 'rgba(0,0,0,0.6)',
  shadow: 'rgba(0,0,0,0.4)',
  inputBg: '#0f172a',
  btnCancelBg: '#334155',
  btnCancelText: '#e2e8f0',
};

const YEAR_RANGE_PRESETS = [
  { label: "Past 5 years", start: 2020, end: 2025 },
  { label: "Past 10 years", start: 2015, end: 2025 },
  { label: "Past 15 years", start: 2010, end: 2025 },
];
const KEYWORD_PRESETS = ["compost", "reuse", "wastewater", "biochar", "fertilizer"];

const FilterSidebar: React.FC<FilterSidebarProps> = ({ visible, filters, onChange, onApply, onCancel, darkMode = false }) => {
  const portalRef = useRef<HTMLDivElement | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [targetDoc, setTargetDoc] = useState<Document | null>(null);

  const theme = darkMode ? darkTheme : lightTheme;

  const handleReset = useCallback(() => {
    onChange("yearStart", "");
    onChange("yearEnd", "");
    onChange("author", "");
    onChange("keywords", "");
  }, [onChange]);

  useEffect(() => {
    let doc: Document = document;
    try {
      if (window.parent && window.parent.document && window.parent.document.body) {
        doc = window.parent.document;
      }
    } catch (e) {}
    setTargetDoc(doc);
    portalRef.current = doc.createElement("div");
    portalRef.current.id = "filter-drawer-portal-root";
    doc.body.appendChild(portalRef.current);
    return () => {
      if (portalRef.current && portalRef.current.parentNode) {
        portalRef.current.parentNode.removeChild(portalRef.current);
      }
      portalRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!visible || !targetDoc) return;
    const onDocClick = (e: MouseEvent) => {
      if (!rootRef.current) return;
      if (!(e.target instanceof Node)) return;
      if (!rootRef.current.contains(e.target)) onCancel();
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    targetDoc.addEventListener("mousedown", onDocClick);
    targetDoc.addEventListener("keydown", onKeyDown);
    return () => {
      targetDoc.removeEventListener("mousedown", onDocClick);
      targetDoc.removeEventListener("keydown", onKeyDown);
    };
  }, [visible, onCancel, targetDoc]);

  if (!portalRef.current) return null;

  // Inline styles
  const wrapperStyle: React.CSSProperties = {
    position: 'fixed',
    inset: 0,
    zIndex: 999999,
    pointerEvents: visible ? 'auto' : 'none',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  };

  const overlayStyle: React.CSSProperties = {
    position: 'fixed',
    inset: 0,
    background: theme.overlay,
    opacity: visible ? 1 : 0,
    transition: 'opacity 200ms ease',
    pointerEvents: visible ? 'auto' : 'none',
    backdropFilter: 'blur(2px)',
  };

  const drawerStyle: React.CSSProperties = {
    position: 'fixed',
    right: 0,
    top: 0,
    bottom: 0,
    width: '340px',
    maxWidth: '90vw',
    background: theme.bg,
    borderLeft: `1px solid ${theme.border}`,
    boxShadow: `-8px 0 32px ${theme.shadow}`,
    transform: visible ? 'translateX(0)' : 'translateX(100%)',
    transition: 'transform 260ms cubic-bezier(.4,0,.2,1)',
    display: 'flex',
    flexDirection: 'column',
    pointerEvents: 'auto',
  };

  const headerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '20px 24px',
    borderBottom: `1px solid ${theme.borderLight}`,
    background: theme.bgHeader,
  };

  const titleStyle: React.CSSProperties = {
    margin: 0,
    fontSize: '18px',
    fontWeight: 600,
    color: theme.text,
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  };

  const closeButtonStyle: React.CSSProperties = {
    border: 'none',
    background: theme.closeBg,
    color: theme.textMuted,
    cursor: 'pointer',
    fontSize: '18px',
    width: '32px',
    height: '32px',
    borderRadius: '8px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 150ms',
  };

  const bodyStyle: React.CSSProperties = {
    padding: '20px 24px',
    overflowY: 'auto',
    flex: '1 1 auto',
  };

  const resetBtnStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    marginBottom: '20px',
    background: 'transparent',
    color: theme.accent,
    border: 'none',
    padding: 0,
    fontSize: '13px',
    fontWeight: 500,
    cursor: 'pointer',
  };

  const sectionStyle: React.CSSProperties = {
    marginBottom: '24px',
  };

  const labelStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '14px',
    fontWeight: 600,
    color: theme.textLabel,
    marginBottom: '10px',
  };

  const chipsContainerStyle: React.CSSProperties = {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    marginBottom: '10px',
  };

  const getChipStyle = (selected: boolean): React.CSSProperties => ({
    background: selected ? theme.chipSelected : theme.chipBg,
    color: selected ? '#fff' : theme.chipText,
    border: `1px solid ${selected ? theme.chipSelected : theme.chipBorder}`,
    borderRadius: '20px',
    padding: '6px 14px',
    fontSize: '13px',
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'all 150ms',
    whiteSpace: 'nowrap',
  });

  const inputStyle: React.CSSProperties = {
    width: '100%',
    height: '42px',
    borderRadius: '10px',
    border: `1px solid ${theme.border}`,
    padding: '0 14px',
    fontSize: '14px',
    color: theme.text,
    background: theme.inputBg,
    transition: 'all 150ms',
    boxSizing: 'border-box',
    outline: 'none',
  };

  const footerStyle: React.CSSProperties = {
    padding: '16px 24px',
    borderTop: `1px solid ${theme.borderLight}`,
    display: 'flex',
    gap: '12px',
    justifyContent: 'flex-end',
    background: theme.bgFooter,
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

  const applyBtnStyle: React.CSSProperties = {
    border: 'none',
    padding: '10px 20px',
    borderRadius: '10px',
    fontWeight: 600,
    fontSize: '14px',
    cursor: 'pointer',
    background: theme.accent,
    color: '#fff',
    transition: 'all 150ms',
  };

  return ReactDOM.createPortal(
    <div style={wrapperStyle}>
      <div style={overlayStyle} onClick={onCancel} aria-hidden />
      <div ref={rootRef} style={drawerStyle} role="dialog" aria-modal="true" tabIndex={-1}>
        <div style={headerStyle}>
          <h3 style={titleStyle}><span style={{ fontSize: '20px' }}>☰</span> Filters</h3>
          <button style={closeButtonStyle} onClick={onCancel} aria-label="Close">✕</button>
        </div>
        <div style={bodyStyle}>
          <button style={resetBtnStyle} type="button" onClick={handleReset}>
            ↺ Reset All
          </button>

          {/* Year Section */}
          <div style={sectionStyle}>
            <div style={labelStyle}>
              <span style={{ fontSize: '16px', opacity: 0.7 }}>📅</span> Year Range
            </div>
            <div style={chipsContainerStyle}>
              {YEAR_RANGE_PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  style={getChipStyle(false)}
                  onClick={() => {
                    onChange("yearStart", preset.start.toString());
                    onChange("yearEnd", preset.end.toString());
                  }}
                  type="button"
                >
                  {preset.label}
                </button>
              ))}
            </div>
            <div style={{ display: 'flex', gap: '10px' }}>
              <input
                style={{ ...inputStyle, flex: 1 }}
                value={filters.yearStart ?? ""}
                onChange={(e) => onChange("yearStart", e.target.value)}
                placeholder="From year"
              />
              <input
                style={{ ...inputStyle, flex: 1 }}
                value={filters.yearEnd ?? ""}
                onChange={(e) => onChange("yearEnd", e.target.value)}
                placeholder="To year"
              />
            </div>
          </div>

          {/* Author Section */}
          <div style={sectionStyle}>
            <div style={labelStyle}>
              <span style={{ fontSize: '16px', opacity: 0.7 }}>👤</span> Author
            </div>
            <input
              style={inputStyle}
              value={filters.author ?? ""}
              onChange={(e) => onChange("author", e.target.value)}
              placeholder="Enter author name..."
            />
          </div>

          {/* Keywords Section */}
          <div style={{ ...sectionStyle, marginBottom: 0 }}>
            <div style={labelStyle}>
              <span style={{ fontSize: '16px', opacity: 0.7 }}>🏷️</span> Keywords
            </div>
            <div style={chipsContainerStyle}>
              {KEYWORD_PRESETS.map((kw) => {
                const kws = filters.keywords ? filters.keywords.split(",").map(s => s.trim().toLowerCase()) : [];
                const isSelected = kws.includes(kw.toLowerCase());
                return (
                  <button
                    key={kw}
                    style={getChipStyle(isSelected)}
                    onClick={() => {
                      if (isSelected) {
                        onChange("keywords", kws.filter(k => k !== kw.toLowerCase()).join(", "));
                      } else {
                        onChange("keywords", [...kws, kw].filter(Boolean).join(", "));
                      }
                    }}
                    type="button"
                  >
                    {kw}
                  </button>
                );
              })}
            </div>
            <input
              style={inputStyle}
              value={filters.keywords ?? ""}
              onChange={(e) => onChange("keywords", e.target.value)}
              placeholder="Or type keywords..."
            />
          </div>
        </div>
        <div style={footerStyle}>
          <button style={cancelBtnStyle} onClick={onCancel}>Cancel</button>
          <button style={applyBtnStyle} onClick={onApply}>Apply Filters</button>
        </div>
      </div>
    </div>,
    portalRef.current
  );
};

export default FilterSidebar;
