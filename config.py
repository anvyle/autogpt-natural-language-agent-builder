"""
Configuration module for AutoGPT Agent Builder.
Handles secrets management for both local development and Streamlit Cloud deployment.
"""

import os
from typing import Optional

# Try to import streamlit for cloud deployment
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Load .env file for local development
from dotenv import load_dotenv
load_dotenv()


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a secret value from Streamlit secrets (cloud) or environment variables (local).
    
    Priority order:
    1. Streamlit secrets (if running on Streamlit Cloud)
    2. Environment variables (for local development)
    3. Default value (if provided)
    
    Args:
        key: The secret key to retrieve
        default: Optional default value if key is not found
        
    Returns:
        The secret value or default if not found
    """
    # Try Streamlit secrets first (for cloud deployment)
    if STREAMLIT_AVAILABLE:
        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    
    # Fall back to environment variables (for local development)
    value = os.getenv(key, default)
    return value


def setup_environment():
    """
    Set up environment variables from secrets.
    This should be called at the start of the application.
    """
    # Set up commonly used environment variables
    api_keys = [
        "GOOGLE_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_TRACING",
        "LANGCHAIN_PROJECT",
    ]
    
    for key in api_keys:
        secret_value = get_secret(key)
        if secret_value and key not in os.environ:
            os.environ[key] = secret_value


def get_google_api_key() -> Optional[str]:
    """Get Google Generative AI API key."""
    return get_secret("GOOGLE_API_KEY")


def get_langchain_api_key() -> Optional[str]:
    """Get LangChain API key."""
    return get_secret("LANGCHAIN_API_KEY")


def is_langchain_tracing_enabled() -> bool:
    """Check if LangChain tracing is enabled."""
    tracing = get_secret("LANGCHAIN_TRACING", "false")
    return tracing.lower() in ("true", "1", "yes")


def get_langchain_project() -> str:
    """Get LangChain project name."""
    return get_secret("LANGCHAIN_PROJECT", "autogpt-agent-builder")


# Initialize environment on module import
setup_environment()

