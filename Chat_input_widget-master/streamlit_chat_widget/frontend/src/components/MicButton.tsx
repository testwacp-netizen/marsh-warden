import React, { useState, useRef } from "react";
import MicIcon from "@mui/icons-material/MicNoneOutlined";
import StopIcon from "@mui/icons-material/Stop";

interface MicButtonProps {
  onSendAudio: (base64: string) => void;
  onRecordingChange?: (isRecording: boolean) => void;
  darkMode?: boolean;
}

const MicButton: React.FC<MicButtonProps> = ({ onSendAudio, onRecordingChange, darkMode = false }) => {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const handleClick = async () => {
    if (isRecording && mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      onRecordingChange && onRecordingChange(false);
      mediaRecorderRef.current = null;
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];

      recorder.ondataavailable = (ev) => {
        if (ev.data.size > 0) audioChunksRef.current.push(ev.data);
      };

      recorder.onstop = () => {
        if (audioChunksRef.current.length === 0) {
          stream.getTracks().forEach((t) => t.stop());
          onRecordingChange && onRecordingChange(false);
          return;
        }
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64String = reader.result as string;
          onSendAudio(base64String);
          audioChunksRef.current = [];
        };
        reader.readAsDataURL(audioBlob);
        stream.getTracks().forEach((t) => t.stop());
        onRecordingChange && onRecordingChange(false);
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      onRecordingChange && onRecordingChange(true);
    } catch (err) {
      console.error("Mic access denied:", err);
    }
  };

  const buttonStyle: React.CSSProperties = darkMode && !isRecording ? {
    background: '#334155',
    color: '#e2e8f0',
  } : {};

  return (
    <button className={`action-btn mic-btn ${isRecording ? "recording" : ""}`} title={isRecording ? "Stop recording" : "Start recording"} onClick={handleClick} style={buttonStyle}>
      {isRecording ? <StopIcon /> : <MicIcon />}
    </button>
  );
};

export default MicButton;
