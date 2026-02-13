import React, { useEffect, useRef, useState } from "react";
import { Streamlit } from "streamlit-component-lib";
import ReactDOM from "react-dom";
import "../ChatInputWidget.css";

interface FilterState {
  year?: string;
  author?: string;
  keywords?: string;
}

interface FilterPopoverProps {
  visible: boolean;
  filters: FilterState;
  onChange: (k: keyof FilterState, v: string) => void;
  onApply: () => void;
  onCancel: () => void;
  anchorOffset?: { left: number; top: number } | null;
}

const FilterPopover: React.FC<FilterPopoverProps> = ({ visible, filters, onChange, onApply, onCancel, anchorOffset }) => {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const container = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Append portal node to the iframe document body only.
    container.current = document.createElement("div");
    container.current.className = "filter-popover-portal";
    document.body.appendChild(container.current);
    return () => {
      if (container.current && container.current.parentNode) container.current.parentNode.removeChild(container.current);
      container.current = null;
    };
  }, []);

  const [placement, setPlacement] = useState<'above' | 'below'>('above');

  useEffect(() => {
    if (!visible) return;
    window.setTimeout(() => {
      rootRef.current?.querySelector("input")?.focus();
    }, 50);
    const onDocClick = (e: MouseEvent) => {
      if (!rootRef.current) return;
      if (!(e.target instanceof Node)) return;
      if (!rootRef.current.contains(e.target)) onCancel();
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [visible, onCancel]);

  // Calculate placement and sizing once the popover is available and we know the anchor.
  useEffect(() => {
    if (!visible || !rootRef.current || !anchorOffset) return;
    const updatePlacement = () => {
      const rect = rootRef.current!.getBoundingClientRect();
      const popHeight = rect.height;
      const anchorY = anchorOffset.top;
      const pad = 12; // spacing between button and popover
      // can we fit above?
      const fitsAbove = anchorY - popHeight - pad > 0;
      setPlacement(fitsAbove ? 'above' : 'below');
      // Compute required height to avoid clipping when the popover is below
      const absoluteBottom = rect.top + window.pageYOffset + rect.height;
      const requiredHeight = Math.ceil(Math.max(absoluteBottom + 24, window.innerHeight));
      try { Streamlit.setFrameHeight(requiredHeight); } catch (e) {}
      // Removed debug logging for placement
    };
    setTimeout(updatePlacement, 10);
    window.addEventListener('resize', updatePlacement);
    return () => window.removeEventListener('resize', updatePlacement);
  }, [visible, anchorOffset]);

  useEffect(() => {
    if (!visible || !rootRef.current) {
      // When the popover is closed, resume default auto-height behavior
      try { Streamlit.setFrameHeight(); } catch (e) {}
      return;
    }
    const updateHeight = () => {
      if (!rootRef.current) return;
      const rect = rootRef.current.getBoundingClientRect();
      const docHeight = document.documentElement?.scrollHeight ?? window.innerHeight;
      const required = Math.ceil(Math.max(rect.bottom + 24, docHeight));
      try {
        Streamlit.setFrameHeight(required);
      } catch (e) {
        // ignore errors
      }
      // Removed debug logging for updateHeight
    };
    updateHeight();
    window.addEventListener("resize", updateHeight);
    return () => {
      window.removeEventListener("resize", updateHeight);
      try { Streamlit.setFrameHeight(); } catch (e) {}
    };
  }, [visible]);

  if (!visible || !container.current) return null;

  const useParent = false; // rendering to parent document was removed to avoid cross-origin complexity

  const style: React.CSSProperties | undefined = anchorOffset
    ? (placement === 'above'
      ? {
          position: "absolute",
          left: `${anchorOffset.left}px`,
          top: `${anchorOffset.top}px`,
          transform: "translateX(-50%) translateY(calc(-100% - 12px))",
        }
      : {
          position: "absolute",
          left: `${anchorOffset.left}px`,
          top: `${anchorOffset.top}px`,
          transform: "translateX(-50%) translateY(12px)",
        }
      )
    : undefined;

  const popoverEl = (
    <div ref={rootRef} className={`filter-popover filter-popover--${placement}`} role="dialog" aria-modal="true" tabIndex={-1} style={style}>
      <div className="filter-left-dot" aria-hidden />
      <div className="filter-form">
        <label className="filter-row">
          <span>Year</span>
          <input value={filters.year ?? ""} onChange={(e) => onChange("year", e.target.value)} placeholder="e.g. 2023" />
        </label>

        <label className="filter-row">
          <span>Author</span>
          <input value={filters.author ?? ""} onChange={(e) => onChange("author", e.target.value)} placeholder="Author name" />
        </label>

        <label className="filter-row">
          <span>Keywords</span>
          <input value={filters.keywords ?? ""} onChange={(e) => onChange("keywords", e.target.value)} placeholder="e.g. compost, reuse" />
        </label>

        <div className="filter-actions">
          <button className="filter-apply" onClick={onApply}>Apply</button>
          <button className="filter-cancel" onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </div>
  );

  return ReactDOM.createPortal(popoverEl, container.current);
};

export default FilterPopover;
