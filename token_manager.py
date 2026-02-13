# token_manager.py
import streamlit as st
import logging
from threading import Lock
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HFTokenRotator:
    def __init__(self, tokens):
        self.tokens = tokens
        self.total_tokens = len(tokens)
        self.current_index = 0
        self.failed_tokens = set()
        self.lock = Lock()
        
        # Track usage and cooldown per token
        self.usage_stats = {
            i: {
                "requests": 0,
                "failures": 0,
                "last_failure": None,
                "cooldown_until": None
            } for i in range(self.total_tokens)
        }
        
        logger.info(f"✅ TokenRotator initialized with {self.total_tokens} tokens")
    
    def get_next_token(self):
        """Get next available token with round-robin and cooldown management"""
        with self.lock:
            attempts = 0
            now = datetime.now()
            
            while attempts < self.total_tokens:
                token_idx = self.current_index
                self.current_index = (self.current_index + 1) % self.total_tokens
                
                # Check if token is in cooldown
                cooldown_until = self.usage_stats[token_idx].get("cooldown_until")
                if cooldown_until and now < cooldown_until:
                    attempts += 1
                    continue
                
                # Skip permanently failed tokens
                if token_idx not in self.failed_tokens:
                    logger.info(f"🔑 Using HF Token #{token_idx + 1}/{self.total_tokens}")
                    return token_idx, self.tokens[token_idx]
                
                attempts += 1
            
            # All tokens are either failed or in cooldown
            # Reset cooldowns and try again
            logger.warning("⚠️ All tokens exhausted or in cooldown, resetting cooldowns...")
            for idx in range(self.total_tokens):
                self.usage_stats[idx]["cooldown_until"] = None
            
            self.failed_tokens.clear()
            return 0, self.tokens[0]
    
    def mark_token_failed(self, token_idx, temporary=True, cooldown_minutes=5):
        """Mark token as failed with optional cooldown"""
        with self.lock:
            if temporary:
                self.usage_stats[token_idx]["failures"] += 1
                self.usage_stats[token_idx]["last_failure"] = datetime.now()
                self.usage_stats[token_idx]["cooldown_until"] = datetime.now() + timedelta(minutes=cooldown_minutes)
                logger.warning(f"⏳ Token #{token_idx + 1} rate limited - cooldown {cooldown_minutes}min")
            else:
                self.failed_tokens.add(token_idx)
                logger.error(f"❌ Token #{token_idx + 1} permanently failed")
    
    def mark_token_success(self, token_idx):
        """Remove token from failed list and clear cooldown on success"""
        with self.lock:
            self.failed_tokens.discard(token_idx)
            self.usage_stats[token_idx]["requests"] += 1
            self.usage_stats[token_idx]["cooldown_until"] = None
    
    def get_stats(self):
        """Get usage statistics"""
        return {
            "total_tokens": self.total_tokens,
            "current_index": self.current_index,
            "failed_count": len(self.failed_tokens),
            "usage": self.usage_stats
        }
    
    def get_available_count(self):
        """Get count of currently available tokens"""
        now = datetime.now()
        available = 0
        for idx in range(self.total_tokens):
            if idx in self.failed_tokens:
                continue
            cooldown = self.usage_stats[idx].get("cooldown_until")
            if cooldown and now < cooldown:
                continue
            available += 1
        return available


def load_hf_tokens_from_secrets():
    """Load all HF tokens from Streamlit secrets - matching your naming convention"""
    tokens = []
    
    print("=" * 70)
    print("🔐 LOADING TOKENS FROM SECRETS")
    print("=" * 70)
    
    # First, try to load HF_TOKEN (your primary token)
    if "HF_TOKEN" in st.secrets:
        tokens.append(st.secrets["HF_TOKEN"])
        print(f"✅  HF_TOKEN: LOADED")
        logger.info("✅ HF_TOKEN loaded")
    else:
        print(f"⚠️  HF_TOKEN: NOT FOUND")
        logger.warning("⚠️ HF_TOKEN not found in secrets")
    
    # Then load backup tokens: HF_BACKUP_TOKEN_1, HF_BACKUP_TOKEN_2, etc.
    for i in range(1, 20):  # Check up to 20 backup tokens
        token_key = f"HF_BACKUP_TOKEN_{i}"
        if token_key in st.secrets:
            token_value = st.secrets[token_key]
            if token_value and token_value.strip():  # Check if not empty
                tokens.append(token_value)
                print(f"✅  {token_key}: LOADED")
                logger.info(f"✅ {token_key} loaded")
            else:
                print(f"⚠️  {token_key}: EMPTY")
        else:
            # Stop checking once we hit a missing token (assumes sequential numbering)
            if i <= 3:  # Only warn for first 3
                print(f"⚠️  {token_key}: NOT FOUND")
            break
    
    print("=" * 70)
    
    if not tokens:
        error_msg = "❌ No HF tokens found in secrets! Please add HF_TOKEN and HF_BACKUP_TOKEN_1, HF_BACKUP_TOKEN_2, etc."
        print(error_msg)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    print(f"✅ Total tokens loaded: {len(tokens)}")
    print("=" * 70)
    logger.info(f"✅ Loaded {len(tokens)} HF tokens from secrets")
    return tokens


@st.cache_resource
def get_token_rotator():
    """Initialize and cache the token rotator"""
    tokens = load_hf_tokens_from_secrets()
    return HFTokenRotator(tokens)