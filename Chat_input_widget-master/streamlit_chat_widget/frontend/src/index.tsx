import React from "react";
import { createRoot } from "react-dom/client";
import ChatInputWidget from "./ChatInputWidget";

const container = document.getElementById("root");
if (container) {
  const root = createRoot(container);
  root.render(
    <React.StrictMode>
      <ChatInputWidget />
    </React.StrictMode>
  );
}
