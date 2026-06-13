"""Central config. Reads secrets from Streamlit when deployed, else from .env locally.

Never hardcode keys. Missing keys are fine — the app degrades gracefully and just
skips the API steps that need them (historical CSV alone still builds a usable model).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load .env if python-dotenv is available (local dev). Harmless if the file is absent.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass


def get_secret(name: str, default: str | None = None) -> str | None:
    """Look up a secret: Streamlit secrets first (cloud), then environment (.env)."""
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


# Convenience accessors
def football_data_key() -> str | None:
    return get_secret("FOOTBALL_DATA_API_KEY")


def api_football_key() -> str | None:
    return get_secret("API_FOOTBALL_KEY")
