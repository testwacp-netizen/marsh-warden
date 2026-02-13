import streamlit as st
from datetime import datetime
import json
import os
from fpdf import FPDF
import hashlib
from uuid import uuid4
import requests
import base64
from streamlit_chat_widget import chat_input_widget
from streamlit_float import float_init
import tiktoken
import logging

# Import RAG pipeline
from cbe_agent import RAGPipeline
from google_auth import check_google_auth
from token_manager import get_token_rotator, HFTokenRotator

# Remove the old HF_TOKEN line and replace with:
# Initialize token rotator
try:
    token_rotator = get_token_rotator()
    # Get first token
    _, HF_TOKEN = token_rotator.get_next_token()
except Exception as e:
    st.error(f"🚨 Failed to initialize token rotator: {e}")
    st.stop()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- CONFIG -------------------
INDEX_FILE = ["pdf_index_enhanced1.pkl", "pdf_index_enhanced2.pkl", "pdf_index_enhanced3.pkl"]

# Access secrets with UPPERCASE keys (matching secrets.toml)
HF_TOKEN = st.secrets["HF_TOKEN"]    # ✅ Fixed - was lowercase
DEEPSEEK_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3.1:novita"

CHAT_HISTORY_DIR = "chat_histories"

# Maximum agent loop iterations
MAX_AGENT_LOOPS = 5

# Professional Color Palette
PRIMARY_COLOR = "#0F766E"
SECONDARY_COLOR = "#06B6D4"
ACCENT_COLOR = "#10B981"
BACKGROUND_LIGHT = "#F0FDFA"
TEXT_PRIMARY = "#0F172A"
TEXT_SECONDARY = "#475569"

# Rest of your code stays exactly the same...

# =============== SESSION STATE ===============

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "model" not in st.session_state:
        st.session_state.model = "DeepSeek"
    if "rag_loaded" not in st.session_state:
        st.session_state.rag_loaded = False
    if "is_switching" not in st.session_state:
        st.session_state.is_switching = False
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if "saved_chat" not in st.session_state:
        st.session_state.saved_chat = None
    if "chat_loaded" not in st.session_state:
        st.session_state.chat_loaded = False

# =============== CUSTOM CSS (same as before, keeping it concise) ===============

def load_custom_css(dark_mode=False):
    # Dark mode color overrides - using Tailwind Slate palette for cohesion
    if dark_mode:
        bg_main = "#0f172a"      # slate-900
        bg_secondary = "#1e293b"  # slate-800
        bg_card = "#1e293b"       # slate-800
        bg_hover = "#334155"      # slate-700
        text_primary = "#f1f5f9"  # slate-100
        text_secondary = "#94a3b8" # slate-400
        text_muted = "#64748b"    # slate-500
        border_color = "#334155"  # slate-700
        accent = "#22d3ee"        # cyan-400
        accent_secondary = "#2dd4bf" # teal-400
    else:
        bg_main = BACKGROUND_LIGHT
        bg_secondary = "#ECFDF5"
        bg_card = "white"
        bg_hover = "#F1F5F9"      # slate-100
        text_primary = TEXT_PRIMARY
        text_secondary = TEXT_SECONDARY
        text_muted = "#64748b"    # slate-500
        border_color = "#E2E8F0"
        accent = PRIMARY_COLOR
        accent_secondary = ACCENT_COLOR
    
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }}
    
    .main {{
        background: {"#0f172a" if dark_mode else f"linear-gradient(135deg, {BACKGROUND_LIGHT} 0%, #ECFDF5 50%, #F0F9FF 100%)"};
        padding: 0;
    }}
    
    .stApp {{
        background: {"#0f172a" if dark_mode else f"linear-gradient(135deg, {BACKGROUND_LIGHT} 0%, #ECFDF5 50%, #F0F9FF 100%)"};
    }}
    
    /* Header Styling */
    .header-container {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 8px 32px rgba(15, 118, 110, 0.15);
        margin-bottom: 2rem;
    }}
    
    .brand-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }}
    
    .brand-icon {{
        font-size: 3rem;
        filter: drop-shadow(0 4px 8px rgba(255,255,255,0.2));
    }}
    
    .header-title {{
        font-size: 2.8rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    .header-subtitle {{
        color: rgba(255, 255, 255, 0.95);
        text-align: center;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.5rem;
        line-height: 1.6;
    }}
    
    .iwmi-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 0.4rem 1rem;
        border-radius: 20px;
        color: white;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }}
    
    /* Chat Messages */
    .stChatMessage {{
        border-radius: 16px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: all 0.2s ease;
        position: relative; /* allow z-index to take effect */
        overflow: visible !important; /* ensure shadow isn't clipped by parent */
        z-index: 1;
    }}
    
    .stChatMessage:hover {{
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }}
    
    [data-testid="stChatMessageContent"] {{
        background-color: transparent !important;
    }}

    /* User Messages */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        background: {"linear-gradient(135deg, #334155 0%, #475569 100%)" if dark_mode else "linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%)"};
        border: {"1px solid #475569" if dark_mode else "none"};
        border-left: 4px solid {"#38bdf8" if dark_mode else SECONDARY_COLOR} !important;
    }}
    
    /* Assistant Messages - more visible in dark mode */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        background: {"linear-gradient(135deg, #475569 0%, #64748b 100%)" if dark_mode else f"linear-gradient(135deg, {BACKGROUND_LIGHT} 0%, #ECFDF5 100%)"};
        border: {"2px solid #94a3b8" if dark_mode else "1px solid #e2e8f0"};
        border-left: 4px solid {"#14b8a6" if dark_mode else ACCENT_COLOR} !important;
        box-shadow: {"0 4px 16px rgba(0,0,0,0.4)" if dark_mode else "0 6px 20px rgba(0,0,0,0.08)"};
    }}
    
    /* Reference Cards */
    .reference-card {{
        background: {"#0f172a" if dark_mode else "white"};
        border: {"1.5px solid #475569" if dark_mode else "1.5px solid #e2e8f0"};
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0;
        transition: all 0.2s ease;
        box-shadow: {"0 2px 8px rgba(0,0,0,0.3)" if dark_mode else "0 1px 3px rgba(0,0,0,0.05)"};
    }}
    
    .reference-card * {{
        color: {text_primary} !important;
    }}
    
    .reference-card p, .reference-card span, .reference-card div {{
        color: {text_secondary} !important;
    }}
    
    .reference-card h1, .reference-card h2, .reference-card h3, 
    .reference-card h4, .reference-card h5, .reference-card strong {{
        color: {text_primary} !important;
    }}

    .reference-card:hover {{
        border-color: {accent};
        box-shadow: {"0 4px 16px rgba(20, 184, 166, 0.3)" if dark_mode else "0 4px 12px rgba(15, 118, 110, 0.1)"};
        transform: translateY(-2px);
    }}
    
    .reference-header {{
        font-weight: 600;
        color: {text_primary};
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .reference-content {{
        color: {text_secondary};
        font-size: 0.88rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }}
    
    .reference-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .badge-table {{ 
        background: {"#422006" if dark_mode else "#FEF3C7"}; 
        color: {"#fbbf24" if dark_mode else "#D97706"} !important;
        border: 1px solid {"#854d0e" if dark_mode else "#FCD34D"};
    }}
    
    .badge-heading {{ 
        background: {"#14532d" if dark_mode else "#D1FAE5"}; 
        color: {"#4ade80" if dark_mode else "#059669"} !important;
        border: 1px solid {"#166534" if dark_mode else "#6EE7B7"};
    }}
    
    .badge-text {{
        background: {"#1e1b4b" if dark_mode else "#E0E7FF"};
        color: {"#a5b4fc" if dark_mode else "#4F46E5"} !important;
        border: 1px solid {"#3730a3" if dark_mode else "#C7D2FE"};
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: {"linear-gradient(180deg, #1e293b 0%, #0f172a 100%)" if dark_mode else "linear-gradient(180deg, #F8FAFC 0%, white 100%)"};
        border-right: 2px solid {border_color};
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 0rem;
    }}
    
    .sidebar-title {{
        font-size: 1.3rem;
        font-weight: 700;
        color: {accent};
        margin-bottom: 0rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .sidebar-section {{
        background: {bg_card};
        padding: 0rem;
        border-radius: 10px;
        margin-bottom: 0rem;
        border: 1px solid {border_color};
    }}
    
    /* Buttons - comprehensive styling */
    .stButton > button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"],
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-primary"],
    button[kind="secondary"],
    button[kind="primary"] {{
        width: 100%;
        background: {"linear-gradient(135deg, #2E8B6E 0%, #3EB489 100%)" if dark_mode else "linear-gradient(135deg, #3EB489 0%, #2E8B6E 100%)"} !important;
        color: white !important;
        border: {"1px solid #3EB489" if dark_mode else "none"} !important;
        padding: 0.5rem 0.75rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        transition: all 0.3s ease !important;
        box-shadow: {"0 4px 12px rgba(62, 180, 137, 0.3)" if dark_mode else "0 4px 12px rgba(62, 180, 137, 0.25)"} !important;
        min-height: 38px !important;
        height: 38px !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }}
    
    /* All button text - covers all possible structures */
    .stButton > button *,
    [data-testid="stBaseButton-secondary"] *,
    [data-testid="stBaseButton-primary"] *,
    [data-testid="baseButton-secondary"] *,
    [data-testid="baseButton-primary"] * {{
        color: white !important;
    }}
    
    /* Sidebar buttons - ensure visibility in dark mode */
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
    section[data-testid="stSidebar"] button[kind="secondary"],
    section[data-testid="stSidebar"] button[kind="primary"] {{
        background: {"linear-gradient(135deg, #2E8B6E 0%, #3EB489 100%)" if dark_mode else "linear-gradient(135deg, #3EB489 0%, #2E8B6E 100%)"} !important;
        color: white !important;
    }}
    
    section[data-testid="stSidebar"] .stButton > button *,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] *,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] * {{
        color: white !important;
    }}
    
    /* Download button styling */
    section[data-testid="stSidebar"] .stDownloadButton > button,
    .stDownloadButton > button,
    [data-testid="stBaseButton-secondary"].stDownloadButton {{
        background: {"#1e293b" if dark_mode else "white"} !important;
        color: {accent} !important;
        border: 2px solid {accent} !important;
    }}
    
    section[data-testid="stSidebar"] .stDownloadButton > button *,
    .stDownloadButton > button * {{
        color: {accent} !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: {"0 6px 20px rgba(62, 180, 137, 0.45)" if dark_mode else "0 6px 20px rgba(62, 180, 137, 0.35)"};
    }}
    
    .stDownloadButton > button {{
        width: 100%;
        background: linear-gradient(135deg, #3EB489 0%, #2E8B6E 100%);
        color: #ffffff !important;
        border: none;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(62, 180, 137, 0.25);
    }}
    
    .stDownloadButton > button p {{
        color: #ffffff !important;
    }}
    
    .stDownloadButton > button:hover {{
        background: linear-gradient(135deg, #2E8B6E 0%, #278F5E 100%);
        color: #ffffff !important;
        box-shadow: 0 6px 20px rgba(62, 180, 137, 0.35);
    }}
    
    .stDownloadButton > button:hover *,
    .stDownloadButton > button:hover p {{
        color: #ffffff !important;
    }}
    
    /* Tooltip styling for dark mode */
    [data-baseweb="tooltip"],
    [data-baseweb="popover"],
    .stTooltipContent,
    .stTooltipHoverTarget + div,
    div[role="tooltip"],
    [data-testid="stTooltipContent"],
    [data-testid="stMarkdownContainer"] + div[role="tooltip"] {{
        background: {"#1e293b" if dark_mode else "#334155"} !important;
        color: {"#f1f5f9" if dark_mode else "white"} !important;
        border: {"1px solid #475569" if dark_mode else "none"} !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }}
    
    [data-baseweb="tooltip"] *,
    [data-baseweb="popover"] *,
    div[role="tooltip"] *,
    [data-testid="stTooltipContent"] * {{
        color: {"#f1f5f9" if dark_mode else "white"} !important;
        background: transparent !important;
    }}
    
    /* Streamlit popover body - the actual content */
    [data-baseweb="popover"] > div {{
        background: {"#1e293b" if dark_mode else "#334155"} !important;
    }}
    
    /* Input Styling */
    .stChatInput {{
        border-radius: 24px;
    }}
    
    .stChatInput > div {{
        border-radius: 24px;
        border: 2px solid {"#334155" if dark_mode else "#E2E8F0"};
        background: {bg_card};
        transition: all 0.2s ease;
    }}
    
    .stChatInput > div:focus-within {{
        border-color: {accent};
        box-shadow: 0 0 0 3px {"rgba(20, 184, 166, 0.2)" if dark_mode else "rgba(15, 118, 110, 0.1)"};
    }}
    
    .stChatInput textarea,
    .stChatInput input {{
        border-radius: 24px !important;
    }}
    
    /* Expander Styling */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] > div:first-child {{
        background: {"linear-gradient(135deg, #1e293b 0%, #334155 100%)" if dark_mode else "linear-gradient(135deg, #F8FAFC 0%, white 100%)"} !important;
        border-radius: 10px;
        font-weight: 600;
        color: {accent} !important;
        border: 1px solid {border_color} !important;
    }}
    
    /* Fix expander summary layout - icon flush with border */
    [data-testid="stExpander"] summary {{
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }}
    
    [data-testid="stExpander"] summary svg {{
        flex-shrink: 0 !important;
        margin: 0 !important;
    }}
    
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    .streamlit-expanderHeader span,
    .streamlit-expanderHeader p {{
        color: {accent} !important;
    }}
    
    [data-testid="stExpander"] svg {{
        color: {accent} !important;
        fill: {accent} !important;
    }}

    .streamlit-expanderHeader:hover,
    [data-testid="stExpander"] summary:hover {{
        background: {"linear-gradient(135deg, #334155 0%, #475569 100%)" if dark_mode else f"linear-gradient(135deg, {BACKGROUND_LIGHT} 0%, #F8FAFC 100%)"} !important;
    }}
    
    /* Expander content area */
    [data-testid="stExpander"] > div[data-testid="stExpanderDetails"],
    [data-testid="stExpander"] > div:last-child {{
        background: {"#1e293b" if dark_mode else "white"} !important;
        border: 1px solid {border_color} !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }}
    
    /* Spinner */
    .stSpinner > div {{
        border-top-color: {PRIMARY_COLOR} !important;
    }}
    
    /* Info Box */
    .info-box {{
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border-left: 4px solid {SECONDARY_COLOR};
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: {TEXT_PRIMARY};
    }}
    
    .warning-box {{
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: {TEXT_PRIMARY};
    }}
    
    /* Session Info */
    .session-info {{
        background: {bg_card};
        padding: 0.75rem 1rem;
        border-radius: 10px;
        font-size: 0.85rem;
        color: {text_muted};
        border: 1px solid {border_color};
    }}
    
    .stat-box {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid {border_color};
    }}
    
    .stat-box:last-child {{
        border-bottom: none;
    }}
    
    .stat-label {{
        color: {text_muted};
        font-weight: 500;
    }}
    
    .stat-value {{
        color: {accent};
        font-weight: 700;
    }}
    
    /* Hide Streamlit Elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    .stDeployButton {{
        display: none !important;
        visibility: hidden !important;
    }}
    
    [data-testid="stSidebarNav"] {{
        display: block !important;
    }}
    
    button[kind="header"] {{
        display: block !important;
        visibility: visible !important;
    }}
    
    [data-testid="collapsedControl"] {{
        display: flex !important;
        visibility: visible !important;
        color: {accent} !important;
        background: {bg_card} !important;
        border: 2px solid {accent} !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        margin: 1rem !important;
        box-shadow: {"0 2px 8px rgba(20, 184, 166, 0.3)" if dark_mode else "0 2px 8px rgba(15, 118, 110, 0.2)"} !important;
    }}
    
    [data-testid="collapsedControl"]:hover {{
        background: {bg_hover} !important;
        transform: scale(1.05);
        box-shadow: {"0 4px 12px rgba(20, 184, 166, 0.4)" if dark_mode else "0 4px 12px rgba(15, 118, 110, 0.3)"} !important;
    }}
    
    header {{
        visibility: visible !important;
    }}
    
    header[data-testid="stHeader"] {{
        background-color: transparent !important;
    }}
    
    section[data-testid="stSidebar"] button[kind="header"] {{
        color: {PRIMARY_COLOR} !important;
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {"#1e293b" if dark_mode else "#F1F5F9"};
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {accent};
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {"#0d9488" if dark_mode else SECONDARY_COLOR};
    }}
    
    /* Profile Card Styling - Google-like */
    .profile-card {{
        background: linear-gradient(135deg, #3EB489 0%, #2E8B6E 100%);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(62, 180, 137, 0.3);
        position: relative;
        overflow: hidden;
    }}
    
    .profile-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ left: -100%; }}
        100% {{ left: 100%; }}
    }}
    
    .profile-image-wrapper {{
        position: relative;
        width: 70px;
        height: 70px;
        margin: 0 auto 0.75rem;
        border: 3px solid white;
        border-radius: 50%;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 100%);
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .profile-initial {{
        font-size: 2rem;
        font-weight: 600;
        color: #3EB489;
        text-transform: uppercase;
        user-select: none;
    }}
    
    .profile-image {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }}
    
    .profile-info h3 {{
        margin: 0.5rem 0 0.25rem 0;
        color: white;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: -0.3px;
    }}
    
    .profile-info p {{
        margin: 0 0 0.75rem 0;
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.75rem;
        font-weight: 500;
        word-break: break-all;
    }}
    
    /* Logout Button Styling */
    div[data-testid="stVerticalBlock"] > div:has(button[key="logout_btn"]) {{
        margin-top: -0.75rem !important;
    }}
    
    button[key="logout_btn"] {{
        width: 100%;
        background: linear-gradient(135deg, #3EB489 0%, #2E8B6E 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.5rem 0.75rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        transition: all 0.3s ease !important;
        box-shadow: none !important;
        font-family: 'Inter', sans-serif !important;
    }}
    
    button[key="logout_btn"]:hover {{
        background: linear-gradient(135deg, #2E8B6E 0%, #278F5E 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(62, 180, 137, 0.35) !important;
    }}
    
    button[key="logout_btn"]:active {{
        transform: translateY(0) !important;
    }}
    
    /* Theme Toggle Button Styling */
    button[key="theme_toggle_btn"] {{
        width: 100%;
        background: {"linear-gradient(135deg, #2E8B6E 0%, #3EB489 100%)" if dark_mode else "linear-gradient(135deg, #3EB489 0%, #2E8B6E 100%)"} !important;
        color: #ffffff !important;
        border: {"1px solid #3EB489" if dark_mode else "none"} !important;
        padding: 0.5rem 0.75rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        transition: all 0.3s ease !important;
        box-shadow: none !important;
        font-family: 'Inter', sans-serif !important;
    }}
    
    button[key="theme_toggle_btn"]:hover {{
        background: {"linear-gradient(135deg, #2E8B6E 0%, #278F5E 100%)" if dark_mode else "linear-gradient(135deg, #2E8B6E 0%, #278F5E 100%)"} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(62, 180, 137, 0.35) !important;
    }}
    
    /* Sidebar Divider */
    .sidebar-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
        margin: 0.5rem 0;
        border: none;
    }}
    
    .control-panel-header {{
        font-size: 0.7rem;
        font-weight: 700;
        color: {TEXT_SECONDARY};
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 0.5rem 0 0.35rem 0;
        margin: 0;
    }}
    
    .section-content {{
        margin-bottom: 0.5rem;
    }}

    .reference-card.reference-highlight {{
        box-shadow: 0 0 0 2px {"#58A6FF" if dark_mode else "#0F766E"}, 0 0 18px {"rgba(88,166,255,0.5)" if dark_mode else "rgba(15,118,110,0.5)"};
        transition: box-shadow 0.3s ease;
    }}

    button[key="auto_download_btn"] {{ display: none !important; }}
    
    /* Dark mode global text */
    {"" if not dark_mode else f'''
    .stMarkdown, .stText, p, span, label, .stSelectbox label, .stRadio label {{
        color: {text_primary} !important;
    }}
    
    [data-testid="stChatMessageContent"] p {{
        color: {text_primary} !important;
    }}
    
    .stSelectbox > div > div {{
        background-color: {bg_card} !important;
        color: {text_primary} !important;
        border-color: {border_color} !important;
    }}
    
    .stTextInput > div > div > input {{
        background-color: {bg_card} !important;
        color: {text_primary} !important;
        border-color: {border_color} !important;
    }}
    '''}
    </style>
    """, unsafe_allow_html=True)


# In-memory store for guest sessions
@st.cache_resource(show_spinner=False)
def _guest_store():
    return {}

# =============== MAIN APP ===============

def main():
    st.set_page_config(
        page_title="Marsh Warden",
        layout="wide",
        page_icon="🌿",
        initial_sidebar_state="expanded"
    )

    # Check for secrets first to avoid crashing with a KeyError
    # Check for secrets first to avoid crashing with a KeyError
def main():
    st.set_page_config(
        page_title="Marsh Warden",
        layout="wide",
        page_icon="🌿",
        initial_sidebar_state="expanded"
    )

    # Check for secrets first to avoid crashing with a KeyError
    # Use the exact secret names as defined in secrets.toml
    required_secrets = ["HF_TOKEN", "client_id", "client_secret", "redirect_uri"]
    missing_secrets = [secret for secret in required_secrets if secret not in st.secrets]
    
    if missing_secrets:
        st.error(f"🚨 Missing Secrets: {', '.join(missing_secrets)}")
        st.warning("Please go to your app settings on Streamlit Cloud and add the missing secrets.")
        st.info("You can copy the template from the `secrets.toml` file in the repository and fill in your values.")
        st.stop()
    
    # Now safely access the secrets (they've been verified to exist)
    HF_TOKEN = st.secrets["HF_TOKEN"]
    # Store it in session state or a global variable for later use
    st.session_state.HF_TOKEN = HF_TOKEN
    
    # 🔐 AUTHENTICATION CHECK
    if not check_google_auth():
        return
    
    init_session_state()

    # For guests, restore conversation from in-memory cache keyed by guest_session
    guest_session_id = st.query_params.get("guest_session")
    if st.session_state.get("guest_authenticated") and guest_session_id:
        store = _guest_store()
        st.session_state.guest_session_id = guest_session_id
        cached = store.get(guest_session_id)
        if cached:
            st.session_state.messages = cached.get("messages", [])
            st.session_state.total_queries = cached.get("total_queries", 0)
            st.session_state.model = cached.get("model", st.session_state.model)
    load_custom_css(st.session_state.dark_mode)
    
    # Inject dark mode class on body for React components to detect
    if st.session_state.dark_mode:
        st.markdown("""
        <script>
            document.body.classList.add('dark-mode');
            document.documentElement.setAttribute('data-theme', 'dark');
        </script>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <script>
            document.body.classList.remove('dark-mode');
            document.documentElement.removeAttribute('data-theme');
        </script>
        """, unsafe_allow_html=True)
    
    # Initialize float for fixed positioning
    float_init()

    # JS helper: smooth scroll + highlight the target card
    st.markdown("""
    <script>
    window.highlightSource = function(targetId) {
        const anchor = document.getElementById(targetId);
        if (!anchor) return;
        anchor.scrollIntoView({behavior: 'smooth', block: 'center'});
        const card = document.getElementById(targetId + '-card');
        if (card) {
            card.classList.add('reference-highlight');
            setTimeout(() => card.classList.remove('reference-highlight'), 1800);
        }
    };
    </script>
    """, unsafe_allow_html=True)
    
    # Get user email
    user_email = st.session_state.google_user.get("email")
    st.session_state.user_email = user_email
    
    # Load chat history from file once per session (value-based check)
    # Skip disk load for guest users (guest state is in-memory cache)
    if not st.session_state.get("chat_loaded", False):
        if not st.session_state.get("guest_authenticated"):
            chat_data = load_chat_history(user_email)
            if chat_data:
                # Keep saved chat metadata available. Only load messages into the live
                # session if the user explicitly restored the conversation (flag file).
                st.session_state.saved_chat = chat_data
                if get_load_on_start(user_email) and chat_data.get("messages"):
                    st.session_state.messages = chat_data.get("messages", [])
                    st.session_state.total_queries = chat_data.get("total_queries", st.session_state.total_queries)
                    st.session_state.model = chat_data.get("model", st.session_state.model)
                else:
                    st.session_state.messages = []
                    st.session_state.total_queries = 0
        st.session_state.chat_loaded = True
    
    # Show user info in sidebar
    if st.session_state.get("google_authenticated") or st.session_state.get("guest_authenticated"):
        user = st.session_state.google_user
        user_name = user.get('name', 'User')
        user_initial = get_user_initial(user_name)
        
        with st.sidebar:
            st.markdown(f"""
            <div class="profile-card">
                <div class="profile-image-wrapper">
                    <img src="{user['picture']}" 
                         alt="{user_name}" 
                         class="profile-image" 
                         loading="lazy" 
                         onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                    <div class="profile-initial" style="display: none;">{user_initial}</div>
                </div>
                <div class="profile-info">
                    <h3>{user_name}</h3>
                    <p>{user['email']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Actual Streamlit logout button
            if st.button("↗️ Sign Out", key="logout_btn", use_container_width=True):
                # Import logout function
                from google_auth import logout
                logout()
            
            # Dark mode toggle
            st.markdown('<div style="margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
            if st.button("🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode", key="theme_toggle_btn", use_container_width=True):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
    
    # Professional loading screen
    if not st.session_state.rag_loaded:
        spinner_text = "🌿 Switching AI Model... Please wait." if st.session_state.is_switching else "🚀 Initializing Marsh Warden... Please wait."
        with st.spinner(spinner_text):
            rag = get_rag_pipeline(st.session_state.model)
            if not rag.load_index():
                st.markdown('<div class="warning-box">⚠️ <strong>Knowledge Base Not Found</strong><br>Please contact your system administrator to build the document index.</div>', unsafe_allow_html=True)
                return
            st.session_state.rag_loaded = True
            st.session_state.rag = rag
            st.session_state.is_switching = False
            st.rerun()
    
    rag = st.session_state.rag
    
    # Get LLM client for agent
    llm_client = get_llm_client(st.session_state.model)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <style>
        .sidebar-divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
            margin: 0.5rem 0;
            border: none;
        }}
        
        .control-panel-header {{
            font-size: 0.7rem;
            font-weight: 700;
            color: {TEXT_SECONDARY};
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 0.5rem 0 0.35rem 0;
            margin: 0;
        }}
        
        .section-content {{
            margin-bottom: 0.5rem;
        }}
        /* Ensure action buttons and inputs in the Past conversations expander use full width */
        .streamlit-expander .stButton>button, .streamlit-expander .stDownloadButton>button, .streamlit-expander .stTextInput>div>input {{
            width: 100% !important;
            box-sizing: border-box;
            white-space: normal !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # Divider after profile
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
       
        
        # Model Selection - without section box
        st.markdown('<div class="section-content">', unsafe_allow_html=True)
        st.session_state.model = "DeepSeek"
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Divider
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<h4 class="control-panel-header">Chat Management</h4>', unsafe_allow_html=True)
        
        # Chat Management - without section box
        st.markdown('<div class="section-content">', unsafe_allow_html=True)
        # Helper: extract first user message to use as a conversation title (minimal, single-line)

        def _first_user_title_from_msgs(msgs: list) -> str | None:
            try:
                for m in msgs:
                    if isinstance(m, dict) and m.get('role') == 'user' and m.get('content'):
                        first_line = str(m.get('content')).strip().splitlines()[0]
                        # Remove obvious UI emojis that we use for archive labels so the title is clean
                        first_line = first_line.replace('📦', '').replace('🎁', '').strip()
                        if len(first_line) > 60:
                            return first_line[:57] + '...'
                        return first_line
            except Exception:
                return None
            return None

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🌿 New", use_container_width=True, help="Start a new conversation"):
                # Archive the current in-memory conversation (best-effort) before resetting.
                archived_in_memory = False
                try:
                    msgs = st.session_state.get("messages", [])
                    if msgs:
                        # Prefer first user question as title, fallback to saved title if present
                        title = _first_user_title_from_msgs(msgs) or None
                        if not title:
                            saved = st.session_state.get("saved_chat")
                            if saved and isinstance(saved, dict):
                                title = saved.get("title")
                        archived = archive_messages(user_email, msgs, st.session_state.get("total_queries", 0), st.session_state.get("model"), title)
                        if archived:
                            archived_in_memory = True
                            st.toast("Conversation archived.", icon="✅")
                except Exception:
                    pass

                # Clear live conversation and start fresh
                st.session_state.messages = []
                st.session_state.total_queries = 0

                # Only archive the on-disk saved session if we did NOT already archive the live messages
                # This avoids producing two archives for the same conversation (one from memory, one from disk).
                try:
                    if not archived_in_memory:
                        archive_current_history(user_email)
                except Exception:
                    pass

                if st.session_state.get("guest_authenticated") and st.session_state.get("guest_session_id"):
                    store = _guest_store()
                    store[st.session_state.guest_session_id] = {
                        "messages": [],
                        "total_queries": 0,
                        "model": st.session_state.model,
                    }
                else:
                    # Ensure a fresh "current" file exists (empty)
                    save_chat_history(user_email, [], 0, st.session_state.model)
                    # Clear any load-on-start flag since user started a new conversation
                    try:
                        clear_load_on_start(user_email)
                    except Exception:
                        pass
                    # Reload saved_chat info for sidebar
                    st.session_state.saved_chat = load_chat_history(user_email)
                st.toast("✨ New conversation started!", icon="🎉")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear all messages"):
                if len(st.session_state.messages) > 0:
                    # Clear live conversation and start fresh
                    st.session_state.messages = []
                    st.session_state.total_queries = 0
                    
                    if st.session_state.get("guest_authenticated") and st.session_state.get("guest_session_id"):
                        store = _guest_store()
                        store[st.session_state.guest_session_id] = {
                            "messages": [],
                            "total_queries": 0,
                            "model": st.session_state.model,
                        }
                    else:
                        # Remove the saved current session (don't create an empty saved conversation)
                        try:
                            delete_chat_history(user_email)
                        except Exception:
                            pass
                        # Clear any load-on-start flag since the current session is now empty
                        try:
                            clear_load_on_start(user_email)
                        except Exception:
                            pass
                        # Refresh saved_chat so the sidebar does not show an empty current session
                        st.session_state.saved_chat = load_chat_history(user_email)
                    st.toast("🗑️ Chat cleared!", icon="✅")
                    st.rerun()
                else:
                    st.toast("⚠️ Chat is already empty!", icon="ℹ️")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Divider
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Past Conversations expander (click item to restore; three-dot shows rename/delete)
        with st.expander("💬 Conversations"):
            # Scoped CSS: make expander's buttons and inputs compact to reduce cramped layout
            st.markdown(
                """
                <style>
                div[data-testid="stExpander"] .stButton button {
                    padding: 4px 8px !important;
                    font-size: 13px !important;
                    height: auto !important;
                    min-width: 36px !important;
                    line-height: 18px !important;
                    border-radius: 8px !important;
                    background-color: rgba(6,182,212,0.06) !important;
                    border: 1px solid rgba(15, 118, 110, 0.08) !important;
                    text-align: left !important;
                    white-space: nowrap !important;
                    overflow: hidden !important;
                    text-overflow: ellipsis !important;
                    color: #064e4a !important;
                }
                div[data-testid="stExpander"] .stTextInput input {
                    font-size: 13px !important;
                    padding: 6px 8px !important;
                }
                .conv-row-sep { margin-top: 4px; margin-bottom: 4px; }
                div[data-testid="stExpander"] .stMarkdown p { margin: 0; padding: 0; }
                </style>
                """,
                unsafe_allow_html=True,
            )
            # Build a flat list: archived sessions first (newest first), then current saved (so archives stack above)
            conv_list = []

            # load archive files and sort by the JSON 'timestamp' field (newest first)
            archive_files = list_archived_histories(user_email) or []
            archives = []
            for p in archive_files:
                a = load_archived_history(p)
                if not a:
                    continue
                # prefer embedded timestamp from the JSON, fallback to file modification time
                ts_raw = a.get("timestamp")
                ts_dt = None
                try:
                    if ts_raw:
                        # handle trailing Z
                        s = str(ts_raw)
                        if s.endswith('Z'):
                            s = s[:-1] + '+00:00'
                        ts_dt = datetime.fromisoformat(s)
                except Exception:
                    ts_dt = None
                if not ts_dt:
                    try:
                        ts_dt = datetime.fromtimestamp(os.path.getmtime(p))
                    except Exception:
                        ts_dt = datetime.now()

                title = (a.get("title") or ts_dt.strftime("%Y-%m-%d %H:%M:%S"))
                preview = (a.get("messages") and a.get("messages")[0].get("content", "")[:160]) or ""
                archives.append({
                    "type": "archive",
                    "id": p,
                    "title": title,
                    "timestamp": a.get("timestamp", ts_dt.isoformat()),
                    "preview": preview,
                    "data": a,
                    "path": p,
                    "_ts_dt": ts_dt
                })

            # sort by parsed timestamp descending (newest first)
            try:
                archives = sorted(archives, key=lambda x: x.get("_ts_dt", datetime.now()), reverse=True)
            except Exception:
                archives = archives

            # append sorted archives to the conv_list
            for a in archives:
                conv_list.append(a)

            # Finally, append the current saved session so that archived items appear above it
            saved = st.session_state.get("saved_chat")
            # Only show current session if we are NOT in "Restored Mode" (flag set)
            # This prevents duplication of the restored chat in the list.
            if saved and not get_load_on_start(user_email):
                conv_list.append({
                    "type": "current",
                    "id": "current",
                    "title": (saved.get("title") or "Current saved session").strip(),
                    "timestamp": saved.get("timestamp", "unknown"),
                    "preview": (saved.get("messages") and saved.get("messages")[0].get("content", "")[:160]) or "",
                    "data": saved
                })

            if not conv_list:
                st.markdown("_No saved or archived conversations found._")
            else:
                # Render each conversation as a row: left = restore button (full width), right = menu button
                def _fmt_ts(ts):
                    if not ts:
                        return ""
                    try:
                        s = str(ts)
                        # handle trailing Z
                        if s.endswith('Z'):
                            s = s[:-1] + '+00:00'
                        dt = datetime.fromisoformat(s)
                        return dt.strftime('%b %d %H:%M')
                    except Exception:
                        try:
                            return str(ts)[:16]
                        except Exception:
                            return str(ts)

                for idx, c in enumerate(conv_list):
                    # Use stable unique key based on conversation id (path or "current")
                    stable_key = hashlib.md5(str(c.get('id', idx)).encode()).hexdigest()[:12]
                    cols = st.columns([7,1,1])
                    col_title, col_space, col_menu = cols[0], cols[1], cols[2]
                    # Compact title (single-line) to avoid multi-line buttons in the sidebar
                    short_title = c.get('title', '')
                    if len(short_title) > 36:
                        short_title = short_title[:33] + '...'
                    display_ts = _fmt_ts(c.get('timestamp', ''))
                    # Sanitize out certain emojis from titles (e.g. remove box/present icons inserted by user)
                    sanitized = short_title.replace('📦', '').replace('🎁', '').strip()
                    display_title = f"{sanitized}"

                    # Title (clickable) and timestamp below
                    with col_title:
                        title_label = display_title
                        if st.button(title_label, key=f"restore_label_{stable_key}", help=(c.get('preview') or ''), use_container_width=True):
                            # Restore messages but remove duplicate entries (preserve order)
                            raw_msgs = c['data'].get('messages', []) or []
                            seen = set()
                            unique_msgs = []
                            for m in raw_msgs:
                                key = (m.get('role'), m.get('content'))
                                if key not in seen:
                                    unique_msgs.append(m)
                                    seen.add(key)

                            st.session_state.messages = unique_msgs
                            st.session_state.total_queries = c['data'].get('total_queries', 0)
                            st.session_state.model = c['data'].get('model', st.session_state.get('model'))

                            # Persist the restored conversation as the 'current' saved session so that
                            # a page refresh resumes the conversation where the user left off.
                            try:
                                if st.session_state.messages:
                                    if st.session_state.get("guest_authenticated") and st.session_state.get("guest_session_id"):
                                        store = _guest_store()
                                        store[st.session_state.guest_session_id] = {
                                            "messages": st.session_state.messages,
                                            "total_queries": st.session_state.total_queries,
                                            "model": st.session_state.model,
                                        }
                                    else:
                                        save_chat_history(user_email, st.session_state.messages, st.session_state.total_queries, st.session_state.model)
                                        # Mark that the restored conversation should be loaded on next session start
                                        try:
                                            set_load_on_start(user_email)
                                        except Exception:
                                            pass
                                        # If the restored conversation has a title, persist it for the saved session
                                        restored_title = c.get('title') or c.get('data', {}).get('title')
                                        if restored_title:
                                            try:
                                                rename_saved_chat(user_email, restored_title)
                                            except Exception:
                                                pass
                                        # Reload saved chat metadata for the sidebar
                                        st.session_state.saved_chat = load_chat_history(user_email)
                            except Exception:
                                pass

                            st.toast("🔁 Conversation restored", icon="✅")
                            st.rerun()
                        # small timestamp below the title
                        st.markdown(f"<div style='font-size:11px;color: #6b7280;margin-top:4px'>{display_ts}</div>", unsafe_allow_html=True)

                    # Three-dot menu toggles inline options - use a separate widget key and a state key
                    menu_btn_key = f"menu_btn_{stable_key}"
                    menu_state_key = f"menu_toggle_{stable_key}"

                    # Ensure there's a default state for the menu (closed)
                    if menu_state_key not in st.session_state:
                        st.session_state[menu_state_key] = False

                    with col_menu:
                        if st.button("⋯", key=menu_btn_key, help="Options: rename / delete"):
                            # If opening this menu, close any other open menu to keep UI tidy
                            will_open = not st.session_state.get(menu_state_key, False)
                            for k in list(st.session_state.keys()):
                                if k.startswith("menu_toggle_") and k != menu_state_key:
                                    st.session_state[k] = False
                            st.session_state[menu_state_key] = will_open

                    # If menu open, show inline rename / delete controls
                    if st.session_state.get(menu_state_key):
                        logger.info(f"[DEBUG-MENU] Menu OPEN for: type={c['type']}, id={c.get('id')}, stable_key={stable_key}")
                        with st.container():
                            # Render rename input and actions inline to avoid vertical stacking
                            action_cols = st.columns([3,1,1])
                            with action_cols[0]:
                                new_title = st.text_input("", key=f"rename_input_{stable_key}", value=c.get('title',''), placeholder="Rename", label_visibility="collapsed")
                            with action_cols[1]:
                                if st.button("✏️", key=f"apply_rename_{stable_key}", help="Rename conversation"):
                                    # Rename current saved or archived
                                    if c['type'] == 'current':
                                        ok = rename_saved_chat(user_email, new_title)
                                        if ok:
                                            st.session_state.saved_chat = load_chat_history(user_email)
                                            st.toast("✏️ Saved session renamed", icon="✅")
                                            # Close any open menus and re-run to refresh display
                                            st.session_state[menu_state_key] = False
                                            st.rerun()
                                    else:
                                        okp = rename_archived_history(c['path'], new_title)
                                        if okp:
                                            st.toast("✏️ Archived session renamed", icon="✅")
                                            # close this menu and refresh
                                            st.session_state[menu_state_key] = False
                                            st.rerun()
                            with action_cols[2]:
                                if st.button("🗑️", key=f"delete_item_{stable_key}", help="Delete conversation"):
                                    logger.info(f"[DEBUG-DELETE] TRIGGERED: type={c['type']}, id={c.get('id')}, path={c.get('path')}, title={c.get('title')}")
                                    if c['type'] == 'current':
                                        logger.info(f"[DEBUG-DELETE] Deleting CURRENT session file")
                                        delete_chat_history(user_email)
                                        st.session_state.saved_chat = None
                                        st.toast("🗑️ Deleted saved session", icon="✅")
                                        st.session_state[menu_state_key] = False
                                        st.rerun()
                                    else:
                                        logger.info(f"[DEBUG-DELETE] Deleting ARCHIVE: {c.get('path')}")
                                        okdel = delete_archived_history(c['path'])
                                        if okdel:
                                            st.toast("🗑️ Deleted archived session", icon="✅");
                                            st.session_state[menu_state_key] = False
                                            st.rerun()
                    # Visual separation between conversation rows
                    st.markdown('<div class="conv-row-sep"></div>', unsafe_allow_html=True)

        # Divider
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        st.markdown('<h4 class="control-panel-header">Export Chat</h4>', unsafe_allow_html=True)
        
        # Export Option - without section box
        st.markdown('<div class="section-content">', unsafe_allow_html=True)
        
        pdf_content = export_conversation_pdf()
        if pdf_content:
            st.download_button(
                label="📕 Download PDF",
                data=pdf_content,
                file_name=f"Marsh Warden_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                help="Download conversation as PDF"
            
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Divider
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        # About Section
        with st.expander("ℹ️ About Marsh Warden "):
            st.markdown("""
            **Marsh Warden** is a specialized AI expert developed to support wetland conservation and environmental policy analysis.

            **Purpose:**
            - Wetland conservation, restoration, and sustainable management
            - Environmental policy analysis and implementation
            - Nature-based solutions and ecosystem services
            - Regulatory frameworks and compliance

            **Target Users:**
            - Environmental policymakers
            - Conservation professionals
            - Regulatory compliance officers
            - Environmental researchers
            - Sustainable development partners
            """)
    
    # Header
    header_images = ["bottu.png", "kokku.png", "Anawilundawa.png"]
    
    # Encode all images
    img_data = []
    for img_path in header_images:
        if os.path.exists(img_path):
            with open(img_path, 'rb') as f:
                img_data.append(base64.b64encode(f.read()).decode())
    
    if len(img_data) >= 3:
        st.markdown(f"""
        <style>
        .header-section {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.95) 0%, rgba(5, 150, 105, 0.95) 100%);
            padding: 1rem 1rem 0.8rem 1rem;
            text-align: center;
            margin: 0;
            position: relative;
        }}
        
        .header-title {{
            font-size: 3rem;
            font-weight: 700;
            color: #ffffff !important;
            margin: 0 0 0.2rem 0;
            letter-spacing: -0.5px;
            line-height: 1;
        }}
        
        .header-subtitle {{
            font-size: 1.1rem;
            font-weight: 400;
            color: #374151;
            margin: 0;
            line-height: 1.2;
        }}
        
        .header-badge {{
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            padding: 0.3rem 0.8rem;
            font-size: 0.75rem;
            color: #ffffff;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            position: absolute;
            bottom: 0.8rem;
            right: 1rem;
        }}
        
        .header-badge::before {{
            content: "🌍";
            font-size: 0.9rem;
        }}
        
        .slideshow-container {{
            width: 100%;
            height: 250px;
            position: relative;
            margin: 0 0 2rem 0;
            border-radius: 0 0 24px 24px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(15, 118, 110, 0.15);
        }}
        
        .slideshow-container img {{
            position: absolute;
            width: 100%;
            height: 250px;
            object-fit: cover;
            opacity: 0;
            animation: slide 12s infinite;
        }}
        
        .slideshow-container img:nth-child(1) {{
            animation-delay: 0s;
        }}
        
        .slideshow-container img:nth-child(2) {{
            animation-delay: 4s;
        }}
        
        .slideshow-container img:nth-child(3) {{
            animation-delay: 8s;
        }}
        
        @keyframes slide {{
            0% {{ opacity: 0; }}
            3% {{ opacity: 1; }}
            33% {{ opacity: 1; }}
            36% {{ opacity: 0; }}
            100% {{ opacity: 0; }}
        }}
        </style>
        
        <div class="header-section">
            <h1 class="header-title">Marsh Warden</h1>
            <p class="header-subtitle">Wetland Information & Conservation Policy support Assistant - Sri Lanka</p>
            <div class="header-badge">Powered by IWMI Research</div>
        </div>
        
        <div class="slideshow-container">
            <img src="data:image/png;base64,{img_data[0]}" alt="Slide 1">
            <img src="data:image/png;base64,{img_data[1]}" alt="Slide 2">
            <img src="data:image/png;base64,{img_data[2]}" alt="Slide 3">
        </div>
        """, unsafe_allow_html=True)
    
    # Welcome message for new users
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="info-box">
            <strong>🌿Welcome Wetland Information & Conservation Policy support Assistant - Sri Lanka</strong><br>
            I'm your AI-powered expert for wetland conservation and environmental policy analysis. 
            Ask me questions about:<br>
            • Wetland conservation strategies and restoration techniques<br>
            • Environmental policy frameworks and regulatory compliance<br>
            • Nature-based solutions and ecosystem services valuation<br>
            • Climate adaptation and mitigation through wetland management<br>
            • Sustainable development and biodiversity conservation policies
        </div>
        """, unsafe_allow_html=True)
    
    # Check if documents are loaded
    if not rag.documents:
        st.markdown('<div class="warning-box">⚠️ <strong>Knowledge Base Empty</strong><br>No documents are currently loaded. Please contact support.</div>', unsafe_allow_html=True)
        return
    
    # Display chat messages from history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            content = message["content"]

            if message["role"] == "assistant":
                # Ensure each assistant message has a unique msg_id
                msg_id = message.get("msg_id", f"msg-{idx}")

                # Fix bullet point rendering
                content = content.replace("•", "-")

                import re
                def repl(m, _msg_id=msg_id):
                    text = m.group(0)
                    def inner_repl(match):
                        label = match.group(0)
                        num = re.findall(r"\d+", label)[0]
                        target = f"{_msg_id}-source-{num}"
                        return f'<a href="#{target}" onclick="window.highlightSource(\'{target}\'); return false;">{label}</a>'
                    return re.sub(r"Source\s+\d+", inner_repl, text)

                # Match patterns like [Source 1] or [Source 1, Source 2] with optional spacing
                content = re.sub(r"\[\s*(?:Source\s+\d+(?:,\s*)?)+\s*\]", repl, content)
                st.markdown(content, unsafe_allow_html=True)
            else:
                st.markdown(content)

            # References block
            if "references" in message and message["references"]:
                msg_id = message.get("msg_id", f"msg-{idx}")

                with st.expander(f"📚 View {len(message['references'])} Source Documents"):
                    for i, doc in enumerate(message["references"], 1):
                        # Handle both dict (from JSON) and Document objects
                        if isinstance(doc, dict):
                            meta = doc.get("metadata", {})
                            text = doc.get("page_content", "")
                        else:
                            meta = doc.metadata
                            text = doc.page_content

                        src = meta.get("source", "Unknown")
                        page = meta.get("page", "?")
                        doc_type = meta.get("type", "text")

                        if doc_type == "table":
                            type_badge = "🔢 TABLE"
                            badge_class = "badge-table"
                        elif doc_type == "heading":
                            type_badge = "📌 HEADING"
                            badge_class = "badge-heading"
                        else:
                            type_badge = "📄 TEXT"
                            badge_class = "badge-text"

                        # Unique ID per message + source index
                        anchor_id = f"{msg_id}-source-{i}"

                        st.markdown(f"""
                        <a id="{anchor_id}"></a>
                        <div class="reference-card highlight-target" id="{anchor_id}-card">
                            <div class="reference-header">
                                <span>{i}. {src} (Page {page})</span>
                                <span class="reference-badge {badge_class}">{type_badge}</span>
                            </div>
                            <div class="reference-content">{text[:300]}...</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Generate PDF data for the widget to use
    pdf_data_b64 = None
    if st.session_state.messages:
        pdf_content = export_conversation_pdf()
        if pdf_content:
            pdf_data_b64 = base64.b64encode(pdf_content).decode()

    # ===== Custom Chat Input Widget (Fixed at Bottom) =====
    footer_container = st.container()
    with footer_container:
        # Use message count as key to reset widget after each send (prevents duplicate re-sends on rerun)
        widget_key = f"chat_widget_{len(st.session_state.messages)}"
        user_input = None
        try:
            user_input = chat_input_widget(
                key=widget_key,
                pdf_data=pdf_data_b64,
                pdf_filename=f"Marsh Warden_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                dark_mode=st.session_state.dark_mode,
                show_suggestions=(False))
        except Exception as e:
            # Log minimal error info and present fallback so the app remains functional
            import traceback, sys
            tb = traceback.format_exc()
            print("chat_input_widget failed:", e, file=sys.stderr)
            print(tb, file=sys.stderr)
            # Show concise, non-sensitive message in the UI and provide the traceback for debugging
            st.error("⚠️ Custom chat input widget failed to initialize. Falling back to the default chat input.")
            # (Optional) show debug info only when running locally or if you decide to expose it
            if os.environ.get("STREAMLIT_SERVER_ENABLE_WS"):  # local/dev heuristic; remove if needed
                st.text_area("Widget traceback (dev only)", tb, height=200)
            # Fallback to a simple text input: allow the app to keep working
            text_fallback = st.chat_input("Start typing here (widget disabled).")
            if text_fallback:
                user_input = {"text": text_fallback}
    
    # Float the container - transparent background, pointer-events: none allows scrolling through it
    footer_container.float("bottom: 0px; background-color: transparent; padding: 10px 0; pointer-events: none;")
    
    # Re-enable pointer events for the widget inside using CSS
    st.markdown("""
    <style>
    [data-testid="stVerticalBlock"] > div:has(iframe[title*="chat_input_widget"]) {
        pointer-events: auto !important;
    }
                
    
    </style>
    """, unsafe_allow_html=True)
    
    # Process user input from custom widget (text or audio)
    prompt = None
    
    if user_input:

        if "text" in user_input and user_input["text"]:
            # Text input from typing
            prompt = user_input["text"]
        elif "audioFile" in user_input:
            # Audio input from recording: accept multiple serialized formats
            raw_audio = user_input.get("audioFile")
            audio_bytes = None
            try:
                if isinstance(raw_audio, (bytes, bytearray)):
                    audio_bytes = bytes(raw_audio)
                elif isinstance(raw_audio, dict):
                    # Uint8Array serialized as {"0":v0,"1":v1,...} - values may be int or str
                    audio_bytes = bytes([int(raw_audio[k]) for k in sorted(raw_audio, key=int)])
                elif isinstance(raw_audio, (list, tuple)):
                    audio_bytes = bytes(raw_audio)
                elif isinstance(raw_audio, str):
                    # data URL: data:audio/wav;base64,<payload>
                    if raw_audio.startswith("data:") and "," in raw_audio:
                        b64 = raw_audio.split(",", 1)[1]
                        audio_bytes = base64.b64decode(b64)
                    else:
                        # attempt base64 decode
                        try:
                            audio_bytes = base64.b64decode(raw_audio)
                        except Exception:
                            audio_bytes = None
                else:
                    audio_bytes = bytes(list(raw_audio))
            except Exception as e:
                st.error(f"❌ Failed to parse audio payload: {e}")
                audio_bytes = None

            if audio_bytes:
                with st.spinner("🎙️ Transcribing audio..."):
                    transcribed_text = transcribe_audio(audio_bytes)
                # find the last "Transcribing audio..." placeholder (should be the one we added)
                status_idx = None
                for i in range(len(st.session_state.messages) - 1, -1, -1):
                    if st.session_state.messages[i].get("content") == "🎙️ Transcribing audio...":
                        status_idx = i
                        break

                if transcribed_text:
                    # update the placeholder message to 'Transcribed: ...'
                    if status_idx is not None:
                        st.session_state.messages[status_idx]["content"] = f"🎙️ Transcribed: {transcribed_text}"
                    # set the prompt so the regular pipeline handles the next steps (append user + fetch assistant)
                    prompt = transcribed_text
                else:
                    # transcription failed - update placeholder to an error message
                    if status_idx is not None:
                        st.session_state.messages[status_idx]["content"] = "⚠️ Transcription failed."
                # persist changes
                save_chat_history(user_email, st.session_state.messages, st.session_state.total_queries, st.session_state.model)

    if prompt:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Persist user message immediately to avoid losing it if processing fails
        if st.session_state.get("guest_authenticated") and st.session_state.get("guest_session_id"):
            store = _guest_store()
            store[st.session_state.guest_session_id] = {
                "messages": st.session_state.messages,
                "total_queries": st.session_state.total_queries,
                "model": st.session_state.model,
            }
        else:
            save_chat_history(user_email, st.session_state.messages, st.session_state.total_queries, st.session_state.model)
        
        # Build conversation history for agent (only user/assistant messages, no tool messages)
        conv_history = []
        for msg in st.session_state.messages[:-1]:  # Exclude current user message
            if msg["role"] in ["user", "assistant"]:
                conv_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Run agent loop
        with st.spinner("🤖 Marsh Warden is thinking..."):
            try:
                answer, retrieved_docs, loop_count = run_agent_loop(
                    user_question=prompt,
                    conversation_history=conv_history,
                    rag_pipeline=rag,
                    llm_client=llm_client,
                    llm_type="deepseek",
                    model_deployment=None
                )
                
                msg_id = f"msg-{len(st.session_state.messages)}"
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "references": retrieved_docs,
                    "msg_id": msg_id
                })
                st.session_state.total_queries += 1
                
                logger.info(f"Agent completed query in {loop_count} iterations")
            
            except Exception as e:
                error_msg = f"⚠️ **Processing Error**\n\nI encountered an issue: `{str(e)}`\n\nPlease try again."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "references": [],
                    "msg_id": f"msg-{len(st.session_state.messages)}"
                })
        
        # Save chat history to file after each message
        if st.session_state.get("guest_authenticated") and st.session_state.get("guest_session_id"):
            store = _guest_store()
            store[st.session_state.guest_session_id] = {
                "messages": st.session_state.messages,
                "total_queries": st.session_state.total_queries,
                "model": st.session_state.model,
            }
        else:
            save_chat_history(user_email, st.session_state.messages, st.session_state.total_queries, st.session_state.model)
        
        # Rerun to display the updated messages
        st.rerun()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0; color: #94A3B8; font-size: 0.85rem;">
        <p>🌱 <strong>Marsh Warden</strong> - Empowering Evidence-Based Decisions</p>
        <p>Developed by IWMI | <a href="https://www.iwmi.cgiar.org">www.iwmi.cgiar.org</a></p>
    </div>
    """, unsafe_allow_html=True)


def get_tool_definitions():
    """Define available tools for the agent"""
    return [
        {
            "type": "function",
            "function": {
                "name": "retrieve_documents",
                "description": "Retrieve relevant documents from the wetland conservation knowledge base to answer questions about wetland protection policies, environmental regulations, conservation acts, panel guidelines, wetland management practices, and sustainable ecosystem preservation. Use this tool when you need authoritative information from policy documents, legal frameworks, regulatory guidelines, and conservation research materials related to wetland ecosystems and their protection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's question or query to search for relevant documents"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of documents to retrieve (default: 8)",
                            "default": 8
                        }
                    },
                    "required": ["question"]
                }
                       }
        }
    ]

def execute_tool(tool_name: str, tool_args: dict, rag_pipeline) -> dict:
    """Execute a tool call and return the result"""
    if tool_name == "retrieve_documents":
        question = tool_args.get("question", "")
        top_k = tool_args.get("top_k", 8)
        
        result = rag_pipeline.retrieve_documents(question, top_k)
        return result
    else:
        return {
            "success": False,
            "message": f"Unknown tool: {tool_name}",
            "documents": [],
            "count": 0
        }

# =============== AGENT LOOP ===============

def run_agent_loop(user_question: str, conversation_history: list, rag_pipeline, llm_client, llm_type: str, model_deployment: str = None) -> tuple:
    """
    Run agent loop with tool calling.
    Returns: (final_answer, retrieved_documents, loop_count)
    """
    tools = get_tool_definitions()
    
    system_prompt = """You are the Marsh Warden - an AI-powered expert specialized in wetland conservation, environmental policy analysis, and sustainable ecosystem management for Sri Lanka.

MANDATORY RESPONSE STRUCTURE:

1. LEGAL HIERARCHY (Priority Order):
   - Primary Authority: Acts/Policies/Strategies containing the specific keyword in title or core focus
   - Secondary Authority: Related NRM instruments governing the activity or site
   - Tertiary Authority: Broad environmental frameworks providing overarching legal power

2. COMPREHENSIVE GUIDANCE (300-400 WORDS):
   - Provide DETAILED, well-structured explanations of rules, restrictions, or requirements
   - Use formatting appropriate to the content:
     * **Tables** for comparative information, regulatory frameworks, or multi-criteria data
     * **Bullet points** for lists of requirements, procedures, or key points
     * **Numbered lists** for sequential steps or hierarchical information
     * **Paragraphs** for explanatory context, background, and nuanced discussion
   - Include relevant context, practical examples, and implementation details
   - Explain implications and practical applications
   - Base strictly on provided documents

3. CITATIONS:
   - Cite every fact using format: `<Act/Policy Name, Section/Page>`
   - No statement without citation

4. FOLLOW-UP:
   - End with ONE proactive question to help user move from knowledge to compliance/action

FORMATTING GUIDELINES:
- Use tables when presenting:
  * Comparative analysis (e.g., different wetland types, permit requirements)
  * Regulatory frameworks with multiple criteria
  * Classification systems or categorizations
- Use bullet points for:
  * Lists of requirements or procedures
  * Key points or highlights
  * Multiple related items
- Use paragraphs for:
  * Conceptual explanations
  * Background context
  * Nuanced discussions
  * Connecting different regulatory aspects

KNOWLEDGE CONSTRAINTS:
- Rely STRICTLY on provided documents only
- If a relevant act exists but is NOT in your library, state: "The [Act Name] is likely the primary authority here, but it is not currently in my reference library."
- Never fabricate or assume content not in your documents

OUTPUT REQUIREMENTS:
- Deliver comprehensive answers: 300-400 words
- Use appropriate formatting (tables, bullets, paragraphs) based on content
- Be thorough while maintaining clarity
- Provide actionable, detailed guidance

Your goal: Provide accurate, hierarchical, citation-backed environmental policy guidance with comprehensive detail for practical compliance and conservation action."
- Never fabricate or assume content not in your documents

OUTPUT REQUIREMENTS:
- Deliver comprehensive answers: 300-400 words
- Use appropriate formatting (tables, bullets, paragraphs) based on content
- Be thorough while maintaining clarity
- Provide actionable, detailed guidance

Your goal: Provide accurate, hierarchical, citation-backed environmental policy guidance with comprehensive detail for practical compliance and conservation action."""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current user question
    messages.append({"role": "user", "content": user_question})
    
    all_retrieved_docs = []
    loop_count = 0
    
    for iteration in range(MAX_AGENT_LOOPS):
        loop_count += 1
        logger.info(f"Agent loop iteration {loop_count}/{MAX_AGENT_LOOPS}")
        
        try:
            headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": DEEPSEEK_MODEL_NAME,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 8000,
                "temperature": 0.1
            }

            r = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)

            if r.status_code != 200:
                logger.error(f"DeepSeek API error: {r.status_code} - {r.text}")
                return f"Sorry, model error: {r.status_code}", [], loop_count

            data = r.json()
            response_message = data["choices"][0]["message"]
            finish_reason = data["choices"][0]["finish_reason"]
            
            # Check if tool calls are needed
            tool_calls = getattr(response_message, 'tool_calls', None) or response_message.get('tool_calls')
            
            if not tool_calls or finish_reason == "stop":
                # No more tool calls, return final answer
                final_content = response_message.content if hasattr(response_message, 'content') else response_message.get('content', '')
                logger.info(f"Agent completed in {loop_count} iterations")
                return final_content, all_retrieved_docs, loop_count
            
            # Add assistant message with tool calls to history
            messages.append({
                "role": "assistant",
                "content": response_message.content if hasattr(response_message, 'content') else response_message.get('content'),
                "tool_calls": tool_calls if isinstance(tool_calls, list) else [tc.__dict__ if hasattr(tc, '__dict__') else tc for tc in tool_calls]
            })
            
            # Execute each tool call
            for tool_call in tool_calls:
                if hasattr(tool_call, 'function'):
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    tool_call_id = tool_call.id
                else:
                    function_name = tool_call['function']['name']
                    function_args = json.loads(tool_call['function']['arguments'])
                    tool_call_id = tool_call['id']
                
                logger.info(f"Executing tool: {function_name} with args: {function_args}")
                
                # Execute tool
                tool_result = execute_tool(function_name, function_args, rag_pipeline)
                
                # Collect retrieved documents
                if tool_result.get("success") and tool_result.get("documents"):
                    all_retrieved_docs.extend(tool_result["documents"])
                
                # Format tool result for LLM
                tool_response_content = json.dumps(tool_result)
                
                # Add tool response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_response_content
                })
            
        except Exception as e:
            logger.error(f"Agent loop error: {e}")
            return f"Sorry, I encountered an error: {str(e)}", all_retrieved_docs, loop_count
    
    # Max iterations reached
    logger.warning(f"Agent reached max iterations ({MAX_AGENT_LOOPS})")
    return "I apologize, but I need more time to process your request. Please try rephrasing your question.", all_retrieved_docs, loop_count

# =============== CHAT HISTORY MANAGEMENT ===============

def get_chat_history_file(email: str) -> str:
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)
    safe_email = hashlib.md5(email.encode()).hexdigest()
    return os.path.join(CHAT_HISTORY_DIR, f"{safe_email}_chat.json")

def _load_on_start_path(email: str) -> str:
    safe_email = hashlib.md5(email.encode("utf-8")).hexdigest()
    return os.path.join(CHAT_HISTORY_DIR, f"{safe_email}_load_on_start")

def set_load_on_start(email: str):
    try:
        p = _load_on_start_path(email)
        with open(p, "w", encoding="utf-8"):
            pass
        return True
    except Exception:
        return False

def clear_load_on_start(email: str):
    try:
        p = _load_on_start_path(email)
        if os.path.exists(p):
            os.remove(p)
        return True
    except Exception:
        return False

def get_load_on_start(email: str) -> bool:
    try:
        return os.path.exists(_load_on_start_path(email))
    except Exception:
        return False

def save_chat_history(email: str, messages: list, total_queries: int, model: str):
    if st.session_state.get("guest_authenticated"):
        return True
    try:
        file_path = get_chat_history_file(email)
        
        serializable_messages = []
        for msg in messages:
            msg_copy = {
                "role": msg["role"],
                "content": msg["content"]
            }
            
            refs = []
            if "references" in msg and msg["references"]:
                for doc in msg["references"]:
                    try:
                        if isinstance(doc, dict):
                            refs.append(doc)
                        else:
                            refs.append({"content": str(doc), "metadata": {}})
                    except Exception:
                        refs.append({"content": "", "metadata": {}})
            msg_copy["references"] = refs
            
            serializable_messages.append(msg_copy)
        
        existing_title = None
        try:
            existing_title = st.session_state.get("saved_chat", {}).get("title") if isinstance(st.session_state.get("saved_chat"), dict) else None
        except Exception:
            existing_title = None

        if not existing_title and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    existing_title = existing.get("title")
            except Exception:
                existing_title = None

        chat_data = {
            "user_email": email,
            "timestamp": datetime.now().isoformat(),
            "messages": serializable_messages,
            "total_queries": total_queries,
            "model": model,
        }
        if existing_title and messages:
            chat_data["title"] = existing_title
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        logger.warning(f"Failed to save chat history: {e}")
        return False

def load_chat_history(email: str) -> dict:
    try:
        file_path = get_chat_history_file(email)
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            chat_data = json.load(f)
        
        return chat_data
    except Exception as e:
        logger.warning(f"Failed to load chat history: {e}")
        return None

def delete_chat_history(email: str) -> bool:
    try:
        file_path = get_chat_history_file(email)
        logger.info(f"[DEBUG-SAVED-DELETE] Attempting to delete saved chat: {file_path}")
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[DEBUG-SAVED-DELETE] Deleted saved chat: {file_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to delete chat history: {e}")
        return False

def _archive_filename_for(email: str, timestamp: str, title: str | None = None) -> str:
    safe_email = hashlib.md5(email.encode("utf-8")).hexdigest()
    uid = uuid4().hex[:8]
    if title:
        slug = "".join(c if c.isalnum() else "_" for c in title)[:40]
        return os.path.join(CHAT_HISTORY_DIR, f"{safe_email}_archive_{timestamp}_{slug}_{uid}.json")
    return os.path.join(CHAT_HISTORY_DIR, f"{safe_email}_archive_{timestamp}_{uid}.json")

def archive_current_history(email: str) -> str | None:
    try:
        current = get_chat_history_file(email)
        if not os.path.exists(current):
            return None
        try:
            with open(current, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        # Don't archive empty conversations or those with no assistant response
        msgs = data.get("messages", [])
        if not msgs:
            return None
        if not any(m.get("role") == "assistant" for m in msgs):
            return None

        # Check against last archive to avoid duplicates
        archives = list_archived_histories(email)
        if archives:
            last_archive = load_archived_history(archives[0])
            if last_archive:
                last_msgs = last_archive.get("messages", [])
                # Simple comparison of message count and content
                if len(msgs) == len(last_msgs):
                    if msgs and last_msgs:
                        if msgs[-1].get("content") == last_msgs[-1].get("content"):
                             if msgs[0].get("content") == last_msgs[0].get("content"):
                                 return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = _archive_filename_for(email, ts)
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return archive_path
    except Exception as e:
        logger.warning(f"Failed to archive chat history: {e}")
        return None

def archive_messages(email: str, messages: list, total_queries: int = 0, model: str = None, title: str | None = None) -> str | None:
    try:
        if not messages:
            return None
            
        # Don't archive if there are no assistant responses (e.g. user just typed "Hi" and cleared)
        if not any(m.get("role") == "assistant" for m in messages):
            return None
        
        # Check if this exact conversation is already the most recent archive to prevent duplicates
        # (e.g. user restores a chat, then immediately clicks New without adding messages)
        archives = list_archived_histories(email)
        if archives:
            last_archive = load_archived_history(archives[0])
            if last_archive:
                last_msgs = last_archive.get("messages", [])
                # Simple length check first
                if len(messages) == len(last_msgs):
                    # Check content of last message to be reasonably sure
                    if messages and last_msgs:
                        if messages[-1].get("content") == last_msgs[-1].get("content"):
                            # Check first message too
                            if messages[0].get("content") == last_msgs[0].get("content"):
                                return None

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = _archive_filename_for(email, ts, title)
        chat_data = {
            "user_email": email,
            "timestamp": datetime.now().isoformat(),
            "title": title or "",
            "messages": messages,
            "total_queries": total_queries,
            "model": model or st.session_state.get("model")
        }
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        return archive_path
    except Exception as e:
        logger.warning(f"Failed to archive messages: {e}")
        return None

def rename_saved_chat(email: str, new_title: str) -> bool:
    try:
        path = get_chat_history_file(email)
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["title"] = new_title
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.warning(f"Failed to rename saved chat: {e}")
        return False

def list_archived_histories(email: str) -> list:
    try:
        safe = hashlib.md5(email.encode("utf-8")).hexdigest()
        files = []
        if os.path.exists(CHAT_HISTORY_DIR):
            for fn in os.listdir(CHAT_HISTORY_DIR):
                if fn.startswith(f"{safe}_archive_") and fn.endswith(".json"):
                    files.append(os.path.join(CHAT_HISTORY_DIR, fn))
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files
    except Exception as e:
        print(f"Error listing archives: {e}")
        return []

def load_archived_history(path: str) -> dict | None:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading archive {path}: {e}")
        return None

def delete_archived_history(path: str) -> bool:
    try:
        logger.info(f"[DEBUG-ARCH-DELETE] Attempting to delete archive: {path}")
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"[DEBUG-ARCH-DELETE] Deleted archive: {path}")
            return True
        logger.info(f"[DEBUG-ARCH-DELETE] Archive not found: {path}")
        return False
    except Exception as e:
        logger.error(f"Error deleting archive {path}: {e}")
        return False

def rename_archived_history(path: str, new_title: str) -> str | None:
    try:
        if not os.path.exists(path):
            return None
        basename = os.path.basename(path)
        parts = basename.split("_archive_")
        if len(parts) < 2:
            return None
        prefix = parts[0]
        ts = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y%m%d_%H%M%S")
        slug = "".join(c if c.isalnum() else "_" for c in new_title)[:60]
        new_path = os.path.join(CHAT_HISTORY_DIR, f"{prefix}_archive_{ts}_{slug}.json")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None

        if data is not None:
            try:
                data["title"] = new_title
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        if os.path.exists(new_path):
            try:
                alt_new_path = os.path.join(CHAT_HISTORY_DIR, f"{prefix}_archive_{ts}_{slug}_{uuid4().hex[:6]}.json")
                os.replace(path, alt_new_path)
                return alt_new_path
            except Exception:
                return path
        else:
            try:
                os.replace(path, new_path)
                return new_path
            except Exception:
                return path
    except Exception as e:
        print(f"Error renaming archive {path}: {e}")
        return None



def get_user_initial(name: str) -> str:
    if name:
        return name[0].upper()
    return "U"

def transcribe_audio(audio_bytes):
    if not audio_bytes:
        return None
        
    API_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "audio/wav" 
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=audio_bytes)
        
        if response.status_code == 200:
            result = response.json()
            return result.get("text", "").strip()
        else:
            st.error(f"Transcription failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to transcription service: {e}")
        return None

def clean_text_for_pdf(text):
    replacements = {
        '\u201c': '"',
        '\u201d': '"',
        '\u2018': "'",
        '\u2019': "'",
        '\u2013': '-',
        '\u2014': '--',
        '\u2026': '...',
        '\u2022': '*',
        '\u00a0': ' ',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text.encode('latin-1', errors='ignore').decode('latin-1')

def export_conversation_pdf():
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 20)
        pdf.set_text_color(15, 118, 110)
        pdf.cell(0, 10, 'Marsh Warden Conversation Export', 0, 1, 'C')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(0, 6, f'User: {st.session_state.get("user_email", "Unknown")}', 0, 1)
        pdf.cell(0, 6, f'Date: {datetime.now().strftime("%B %d, 2023 at %I:%M %p")}', 0, 1)
        pdf.cell(0, 6, f'Model: {st.session_state.model}', 0, 1)
        pdf.cell(0, 6, f'Total Queries: {st.session_state.total_queries}', 0, 1)
        pdf.ln(10)
        
        for i, msg in enumerate(st.session_state.messages, 1):
            pdf.set_font('Arial', 'B', 12)
            pdf.set_text_color(15, 118, 110)
            role_text = f"User (Message {i})" if msg["role"] == "user" else f"Marsh Warden (Message {i})"
            pdf.cell(0, 8, role_text, 0, 1)
            
            pdf.set_font('Arial', '', 10)
            pdf.set_text_color(0, 0, 0)
            cleaned_content = clean_text_for_pdf(msg['content'])
            pdf.multi_cell(0, 6, cleaned_content)
            pdf.ln(3)
            
            if "references" in msg and msg["references"]:
                pdf.set_font('Arial', 'I', 9)
                pdf.set_text_color(71, 85, 105)
                pdf.cell(0, 6, f'Sources: {len(msg["references"])} documents referenced', 0, 1)
                
                for j, doc in enumerate(msg["references"][:3], 1):
                    if isinstance(doc, dict):
                        src = clean_text_for_pdf(doc.get("metadata", {}).get("source", "Unknown"))
                        page = doc.get("metadata", {}).get("page", "?")
                    else:
                        src = clean_text_for_pdf(doc.get("source", "Unknown"))
                        page = doc.get("page", "?")
                    
                    pdf.set_font('Arial', '', 8)
                    pdf.cell(0, 5, f'  {j}. {src} (Page {page})', 0, 1)
                
                pdf.ln(2)
            
            pdf.ln(5)
        
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 8)
        pdf.set_text_color(107, 114, 128)
        pdf.multi_cell(0, 5, 'Marsh Warden - Wetland Information & Conservation Policy support Assistant - Sri Lanka\nDeveloped by International Water Management Institute (IWMI)')
        
        pdf_output = pdf.output(dest='S')
        if isinstance(pdf_output, str):
            pdf_output = pdf_output.encode('latin-1')
        return pdf_output
    
    except Exception as e:
        st.error(f"❌ PDF generation failed: {e}")
        return None


# =============== RAG PIPELINE ===============

@st.cache_resource(show_spinner=False)
def get_rag_pipeline(selected_model: str):
    params = {
        "llm_type": "deepseek",
        "hf_token": HF_TOKEN,
        "deepseek_url": DEEPSEEK_API_URL,
        "deepseek_model": DEEPSEEK_MODEL_NAME,
    }
    return RAGPipeline(
        pdf_folder="",  # Not needed for runtime, only for building the index
        index_file=INDEX_FILE,
        model_params=params,
    )

@st.cache_resource(show_spinner=False)
def get_llm_client(selected_model: str):
    """Get LLM client for agent loop"""
    return None
if __name__ == "__main__":
    main()
