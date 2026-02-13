import streamlit.components.v1 as components
import os

_RELEASE = True

if _RELEASE:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend", "build")
    _component_func = components.declare_component("chat_input_widget", path=build_dir)
else:
    _component_func = components.declare_component("chat_input_widget", url="http://localhost:3000")

def chat_input_widget(key=None, pdf_data=None, pdf_filename="conversation.pdf", dark_mode=False, show_suggestions=False):
    """
    Custom chat input widget with text, audio, and PDF download support.
    
    Args:
        key: Unique key for the component
        pdf_data: Base64-encoded PDF data for download
        pdf_filename: Filename for the downloaded PDF
        dark_mode: Whether dark mode is enabled
        show_suggestions: Whether to show suggestion chips (for empty chat)
    
    Returns:
        dict with user input (text, audioFile, or download action)
    """
    component_value = _component_func(
        key=key,
        pdf_data=pdf_data,
        pdf_filename=pdf_filename,
        dark_mode=dark_mode,
        show_suggestions=show_suggestions,
        default=None
    )
    return component_value