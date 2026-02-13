import React from "react";
import SendIcon from "@mui/icons-material/ArrowUpward";

interface SendButtonProps {
  active: boolean;
  onClick: () => void;
  darkMode?: boolean;
}

const SendButton: React.FC<SendButtonProps> = ({ active, onClick, darkMode = false }) => {
  const style: React.CSSProperties = darkMode && !active ? {
    background: '#334155',
    color: '#64748b',
  } : {};

  return (
    <button className={`send-btn ${active ? "active" : ""}`} onClick={onClick} disabled={!active} style={style}>
      <SendIcon />
    </button>
  );
};

export default SendButton;
