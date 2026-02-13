import React from "react";

interface InputFieldProps {
  value: string;
  onChange: (v: string) => void;
  onKeyPress: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  placeholder?: string;
  darkMode?: boolean;
}

const InputField: React.FC<InputFieldProps> = ({ value, onChange, onKeyPress, placeholder, darkMode = false }) => {
  const style: React.CSSProperties = darkMode ? {
    background: 'transparent',
    color: '#f1f5f9',
    border: 'none',
  } : {
    background: 'transparent',
  };

  return (
    <input
      type="text"
      className="chat-input-field"
      placeholder={placeholder}
      value={value}
      onKeyDown={onKeyPress}
      onChange={(e) => onChange((e.target as HTMLInputElement).value)}
      style={style}
    />
  );
};

export default InputField;
