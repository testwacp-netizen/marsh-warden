import streamlit as st
import requests
from urllib.parse import urlencode
from uuid import uuid4
import time
import json
import os

# Color palette
PRIMARY_COLOR = "#0F766E"      # Teal
SECONDARY_COLOR = "#06B6D4"    # Cyan
ACCENT_COLOR = "#10B981"       # Emerald
BACKGROUND_LIGHT = "#F0FDFA"   # Light teal
TEXT_PRIMARY = "#0F172A"       # Slate
TEXT_SECONDARY = "#475569"     # Slate gray

# File to store tokens persistently
TOKEN_STORAGE_FILE = ".streamlit_auth_tokens.json"

class GoogleOAuth:
    def __init__(self):
        self.client_id = st.secrets["client_id"]
        self.client_secret = st.secrets["client_secret"]
        self.redirect_uri = st.secrets.get("redirect_uri", "http://localhost:8501/")
        
        self.scope = (
            "https://www.googleapis.com/auth/userinfo.email "
            "https://www.googleapis.com/auth/userinfo.profile"
        )
        
    def get_authorization_url(self):
        params = {
            "client_id": self.client_id,
            "scope": self.scope,
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent",
            "redirect_uri": self.redirect_uri,
        }
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    
    def get_tokens(self, code):
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri,
        }
        # Debug: Print the redirect_uri being used
        print(f"DEBUG: Using redirect_uri: {self.redirect_uri}")
        
        r = requests.post(token_url, data=data)
        if r.status_code == 200:
            tokens = r.json()
            if 'expires_in' in tokens:
                tokens['expires_at'] = time.time() + tokens['expires_in']
            else:
                tokens['expires_at'] = time.time() + (2 * 60 * 60)
            return tokens
        
        # Debug: Print error details
        print(f"DEBUG: Token error - Status: {r.status_code}, Response: {r.text}")
        st.error(f"Failed to get tokens: {r.text}")
        return None
    
    def get_user_info(self, access_token):
        userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
        headers = {"Authorization": f"Bearer {access_token}"}
        r = requests.get(userinfo_url, headers=headers)
        if r.status_code == 200:
            return r.json()
        st.error(f"Failed to get user info: {r.text}")
        return None
    
    def refresh_access_token(self, refresh_token):
        """Refresh the access token using refresh token"""
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        r = requests.post(token_url, data=data)
        if r.status_code == 200:
            tokens = r.json()
            tokens['refresh_token'] = refresh_token
            if 'expires_in' in tokens:
                tokens['expires_at'] = time.time() + tokens['expires_in']
            return tokens
        return None

# ==================== PERSISTENT STORAGE FUNCTIONS ====================
import hashlib

def _token_file_for_email(email: str) -> str:
    """Create a per-user token filename (hash to avoid special chars)."""
    h = hashlib.md5(email.encode("utf-8")).hexdigest()
    return f".streamlit_auth_tokens_{h}.json"

def save_tokens_to_file(tokens, user_info):
    """Save tokens and user info to a per-user JSON file."""
    try:
        email = user_info.get("email")
        if not email:
            return False
        filename = _token_file_for_email(email)

        data = {
            "tokens": tokens,
            "user_info": user_info,
            "saved_at": time.time(),
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving tokens: {e}")
        return False
    

def load_tokens_from_file():
    """
    Load tokens ONLY if the URL contains a session ID matching a saved file.
    """
    try:
        # 1. Check if we have a session ID in the URL
        # st.query_params behaves like a dict in newer Streamlit
        session_id = st.query_params.get("session")
        
        if not session_id:
            return None # No ID in URL -> Guest/New User -> Show Login
            
        # 2. Construct filename from the URL ID
        filename = f".streamlit_auth_tokens_{session_id}.json"
        
        if not os.path.exists(filename):
            return None # ID exists but file is gone -> Show Login

        # 3. Load the specific file for this user
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokens = data.get("tokens")
        user_info = data.get("user_info")
        
        if not tokens or not user_info:
            return None

        # 4. Check expiry and refresh if needed
        expires_at = tokens.get("expires_at")
        if expires_at and time.time() > expires_at:
            google_oauth = GoogleOAuth()
            refresh_token = tokens.get("refresh_token")

            if refresh_token:
                new_tokens = google_oauth.refresh_access_token(refresh_token)
                if new_tokens:
                    new_tokens["refresh_token"] = refresh_token
                    save_tokens_to_file(new_tokens, user_info)
                    return {"tokens": new_tokens, "user_info": user_info}
            return None

        return {"tokens": tokens, "user_info": user_info}

    except Exception as e:
        print(f"Error loading tokens: {e}")
        return None

def delete_tokens_from_file():
    """Delete stored tokens for the currently logged-in user when logging out."""
    try:
        user = st.session_state.get("google_user")
        if not user:
            return False
        email = user.get("email")
        if not email:
            return False

        filename = _token_file_for_email(email)
        if os.path.exists(filename):
            os.remove(filename)
        return True
    except Exception as e:
        print(f"Error deleting tokens: {e}")
        return False

#def save_tokens_to_file(tokens, user_info):
    """Save tokens and user info to a JSON file"""
    # try:
    #     data = {
    #         "tokens": tokens,
    #         "user_info": user_info,
    #         "saved_at": time.time()
    #     }
    #     with open(TOKEN_STORAGE_FILE, "w") as f:
    #         json.dump(data, f)
    #     return True
    # except Exception as e:
    #     print(f"Error saving tokens: {e}")
        # return False

#def load_tokens_from_file():
    """Load tokens and user info from file if they exist and are valid"""
    # try:
    #     if not os.path.exists(TOKEN_STORAGE_FILE):
    #         return None
        
    #     with open(TOKEN_STORAGE_FILE, "r") as f:
    #         data = json.load(f)
        
    #     tokens = data.get("tokens")
    #     user_info = data.get("user_info")
        
    #     if not tokens or not user_info:
    #         return None
        
    #     # Check if tokens are expired
    #     expires_at = tokens.get("expires_at")
    #     if expires_at and time.time() > expires_at:
    #         google_oauth = GoogleOAuth()
    #         refresh_token = tokens.get("refresh_token")
            
    #         if refresh_token:
    #             new_tokens = google_oauth.refresh_access_token(refresh_token)
    #             if new_tokens:
    #                 new_tokens['refresh_token'] = refresh_token
    #                 save_tokens_to_file(new_tokens, user_info)
    #                 return {"tokens": new_tokens, "user_info": user_info}
            
    #         return None
        
    #     return {"tokens": tokens, "user_info": user_info}
    
    # except Exception as e:
    #     print(f"Error loading tokens: {e}")
    #     return None

# def delete_tokens_from_file():
#     """Delete stored tokens when logging out"""
#     try:
#         if os.path.exists(TOKEN_STORAGE_FILE):
#             os.remove(TOKEN_STORAGE_FILE)
#         return True
#     except Exception as e:
#         print(f"Error deleting tokens: {e}")
#         return False

# ==================== AUTH FUNCTIONS ====================

def logout():
    """Enhanced logout - clear all session data and stored tokens"""
    keys_to_clear = [
        'google_authenticated', 'guest_authenticated', 'google_user', 'google_access_token',
        'google_refresh_token', 'session_start_time', 'token_expires_at',
        'messages', 'total_queries', 'chat_loaded' # Added chat keys
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    delete_tokens_from_file()
    
    # Clear URL params to prevent auto-login
    st.query_params.clear()
    
    st.rerun()

def check_google_auth():
    """Check if user is authenticated with Google - with persistent storage"""
    google_oauth = GoogleOAuth()
    params = st.query_params

    # DEBUG: Print initialization
    print(f"🔍 DEBUG - Auth Check Started")
    print(f"   redirect_uri loaded: {google_oauth.redirect_uri}")
    print(f"   query_params: {params}")

    # ========== HANDLE GUEST LOGIN ==========
    # Check for persistent guest_session param FIRST (survives refresh)
    guest_session_val = params.get("guest_session")
    if guest_session_val:
        st.session_state.guest_authenticated = True
        st.session_state.guest_session_id = guest_session_val
        st.session_state.google_user = {"name": "Guest", "email": "guest@local", "picture": ""}
        return True
    # Then check session state (within same browser session)
    if st.session_state.get("guest_authenticated"):
        return True
    # Initial guest login - set persistent param
    if params.get("guest") == "1" or params.get("guest") == ["1"]:
        st.session_state.guest_authenticated = True
        st.session_state.google_user = {"name": "Guest", "email": "guest@local", "picture": ""}
        session_id = uuid4().hex
        st.session_state.guest_session_id = session_id
        st.query_params.clear()
        st.query_params["guest_session"] = session_id
        st.rerun()
        return True

    # ========== HANDLE OAUTH CALLBACK ==========
    if "code" in params:
        print(f"✅ OAuth callback detected - code present in params")
        code = params["code"]
        print(f"   code: {code[:30]}...")
        
        # Show loading screen
        st.markdown("""
        <style>
        .auth-loading {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            background: linear-gradient(135deg, #F0FDFA 0%, #CCFBF1 100%);
            gap: 2rem;
        }
        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid #E0E7FF;
            border-top: 4px solid #0F766E;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .auth-text {
            font-size: 1.2rem;
            color: #0F766E;
            font-weight: 600;
        }
        </style>
        <div class="auth-loading">
            <div class="spinner"></div>
            <div class="auth-text">✨ Authenticating with Google...</div>
            <p style="color: #475569;">Please wait while we verify your credentials</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get tokens
        print(f"🔄 Exchanging code for tokens...")
        tokens = google_oauth.get_tokens(code)
        
        if tokens and "access_token" in tokens:
            print(f"✅ Tokens received, fetching user info...")
            user_info = google_oauth.get_user_info(tokens["access_token"])
            
            if user_info:
                print(f"✅ User info received: {user_info.get('email')}")
                
                # Store in session state
                st.session_state.google_authenticated = True
                st.session_state.google_user = user_info
                st.session_state.google_access_token = tokens["access_token"]
                st.session_state.session_start_time = time.time()
                
                if 'refresh_token' in tokens:
                    st.session_state.google_refresh_token = tokens['refresh_token']
                if 'expires_at' in tokens:
                    st.session_state.token_expires_at = tokens['expires_at']
                
                # Save to persistent file
                print(f"💾 Saving tokens to file...")
                save_tokens_to_file(tokens, user_info)
                
                # --- MODIFIED: Set Session ID in URL instead of JS Redirect ---
                print(f"🧹 Setting session ID in URL...")
                
                # Calculate session hash
                email = user_info.get("email")
                session_hash = hashlib.md5(email.encode("utf-8")).hexdigest()

                # Clear query params (removes 'code')
                st.query_params.clear()
                
                # Set the session param so reload works
                st.query_params["session"] = session_hash
                
                # Rerun to update URL
                st.rerun()
                return True
            else:
                print(f"❌ Failed to get user info")
        else:
            print(f"❌ Failed to exchange code for tokens")
        
        st.error("❌ Authentication failed. Please try again.")
        return False

    # ========== CHECK SESSION STATE ==========
    if st.session_state.get("google_authenticated"):
        print(f"✅ User already in session state")
        
        # --- ADDED: Ensure URL has session ID ---
        user_info = st.session_state.get("google_user")
        if user_info:
            email = user_info.get("email")
            session_hash = hashlib.md5(email.encode("utf-8")).hexdigest()
            if st.query_params.get("session") != session_hash:
                st.query_params["session"] = session_hash
        
        session_start = st.session_state.get("session_start_time")
        if session_start and (time.time() - session_start) < (2 * 60 * 60):
            print(f"✅ Session still valid")
            return True
        else:
            print(f"⏰ Session expired - logging out")
            logout()
            return False

    # ========== CHECK PERSISTENT STORAGE ==========
    print(f"🔍 Checking persistent storage...")
    stored_auth = load_tokens_from_file()
    if stored_auth:
        print(f"✅ Found stored tokens - restoring...")
        tokens = stored_auth["tokens"]
        user_info = stored_auth["user_info"]
        
        # Restore session state
        st.session_state.google_authenticated = True
        st.session_state.google_user = user_info
        st.session_state.google_access_token = tokens["access_token"]
        st.session_state.session_start_time = time.time()
        
        if 'refresh_token' in tokens:
            st.session_state.google_refresh_token = tokens['refresh_token']
        if 'expires_at' in tokens:
            st.session_state.token_expires_at = tokens['expires_at']
        
        # --- ADDED: Ensure URL has session ID ---
        email = user_info.get("email")
        session_hash = hashlib.md5(email.encode("utf-8")).hexdigest()
        if st.query_params.get("session") != session_hash:
            st.query_params["session"] = session_hash
        
        print(f"✅ Restored from storage")
        return True

    # ========== SHOW LOGIN PAGE ==========
    print(f"📝 No authentication found - showing login page")
    if not st.session_state.get("google_authenticated"):
        auth_url = google_oauth.get_authorization_url()
        print(f"   Auth URL: {auth_url[:100]}...")
        show_login_page(auth_url)
        return False

    return False

def show_login_page(auth_url):
    """Display the beautiful login page"""
    st.markdown(
        f"""
        <style>
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        @keyframes shimmer {{
            0% {{
                background-position: -1000px 0;
            }}
            100% {{
                background-position: 1000px 0;
            }}
        }}
        
        @keyframes float {{
            0%, 100% {{
                transform: translateY(0px);
            }}
            50% {{
                transform: translateY(-10px);
            }}
        }}
        
        .stApp {{
            background: linear-gradient(135deg, #F0FDFA 0%, #CCFBF1 100%);
        }}
        
        .login-card {{
            max-width: 420px;
            margin: 80px auto;
            padding: 50px 40px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(15, 118, 110, 0.1);
            text-align: center;
            animation: fadeInUp 0.6s ease-out;
            position: relative;
            overflow: hidden;
        }}
        
        .login-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.1), transparent);
            animation: shimmer 3s infinite;
        }}
        
        .login-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(15, 118, 110, 0.15);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .icon-wrapper {{
            display: inline-block;
            font-size: 48px;
            margin-bottom: 20px;
            animation: float 3s ease-in-out infinite;
        }}
        
        .login-title {{
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {SECONDARY_COLOR} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
            animation: fadeInUp 0.6s ease-out 0.2s both;
        }}
        
        .login-subtitle {{
            color: {TEXT_SECONDARY};
            font-size: 16px;
            margin-bottom: 40px;
            animation: fadeInUp 0.6s ease-out 0.4s both;
        }}
        
        .google-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, {ACCENT_COLOR} 100%);
            color: white;
            padding: 14px 36px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 12px;
            text-decoration: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(15, 118, 110, 0.3);
            animation: fadeInUp 0.6s ease-out 0.6s both;
        }}
        
        .google-btn::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s;
        }}
        
        .google-btn:hover::before {{
            left: 100%;
        }}
        
        .google-btn:hover {{
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 6px 25px rgba(15, 118, 110, 0.4);
        }}
        
        .google-btn:active {{
            transform: translateY(0) scale(0.98);
        }}
        
        .google-icon {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
        }}
        
        .security-badge {{
            margin-top: 30px;
            padding: 12px 20px;
            background: {BACKGROUND_LIGHT};
            border-radius: 10px;
            font-size: 13px;
            color: {TEXT_SECONDARY};
            animation: fadeInUp 0.6s ease-out 0.8s both;
        }}
        
        .security-icon {{
            color: {ACCENT_COLOR};
            margin-right: 6px;
        }}
        
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        .stDeployButton {{display: none !important;}}
        </style>
        <div class="login-card">
            <div class="icon-wrapper">🔐</div>
            <div class="login-title">Marsh Warden</div>
            <div class="login-subtitle">Wetland Information & Conservation Policy support Assistant - Sri Lanka</div>
            <a href="{auth_url}" class="google-btn">
                <svg class="google-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path fill="#fff" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                    <path fill="#fff" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                    <path fill="#fff" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                    <path fill="#fff" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                </svg>
                Sign in with Google
            </a>
            <a href="?guest=1" style="display:block; margin-top:20px; color:{TEXT_SECONDARY}; font-size:14px; text-decoration:none;">Login as Guest</a>
            <div style="font-size:11px; color:#94a3b8; margin-top:4px;">Guest sessions are temporary</div>
            <div class="security-badge">
                <span class="security-icon">🛡️</span>
                Secure OAuth 2.0 Authentication • 2-Hour Session
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
