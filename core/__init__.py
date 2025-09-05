# core/__init__.py
# Mark 'core' as a package and re-export the single public entrypoint
# used by the Streamlit app.

from .data_store import load_store

__all__ = ["load_store"]
