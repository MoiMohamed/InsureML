"""
utils/data_loader.py
────────────────────
Centralised data loading logic with multiple fallbacks.

WHY A SEPARATE MODULE?
  Streamlit's @st.cache_data needs to live close to where the function is defined.
  By isolating it here, we avoid re-triggering the cache when other modules change,
  and we keep app.py clean.
"""

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

TARGET_COL = "charges"

DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/stedy/"
    "Machine-Learning-with-R-datasets/master/insurance.csv"
)
LOCAL_PATHS = ["data/insurance.csv", "insurance.csv"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lowercase, replace spaces with underscores."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset(uploaded_csv: io.BytesIO | None = None) -> pd.DataFrame:
    """
    Load the insurance dataset from multiple sources (in priority order):
      1. User-uploaded file via Streamlit sidebar
      2. Local CSV paths (data/insurance.csv or insurance.csv)
      3. KaggleHub (if auth is configured)
      4. Public GitHub URL fallback

    Returns a normalized DataFrame (columns lowercased, spaces → underscores).
    """
    # ── 1. User upload ──────────────────────────────────────────────────────
    if uploaded_csv is not None:
        return _normalize_columns(pd.read_csv(uploaded_csv))

    # ── 2. Local file ───────────────────────────────────────────────────────
    for path in LOCAL_PATHS:
        if os.path.exists(path):
            return _normalize_columns(pd.read_csv(path))

    # ── 3. KaggleHub (optional; works when KAGGLE_USERNAME/KEY are set) ─────
    try:
        import kagglehub  # type: ignore

        dataset_dir = kagglehub.dataset_download("mirichoi0218/insurance")
        csv_files = list(Path(dataset_dir).glob("*.csv"))
        if csv_files:
            return _normalize_columns(pd.read_csv(csv_files[0]))
    except Exception:
        pass

    # ── 4. Public URL fallback ───────────────────────────────────────────────
    try:
        return _normalize_columns(pd.read_csv(DEFAULT_DATA_URL))
    except Exception as exc:
        raise RuntimeError(
            "Could not load dataset. "
            "Place insurance.csv in the project root or upload it from the sidebar."
        ) from exc
