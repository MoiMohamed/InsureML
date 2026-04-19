"""
Insurance Cost Intelligence — Main Entry Point
-----------------------------------------------
All page routing lives here. Each page is its own module under /pages/.
Data loading and model training live in /utils/.
"""

import warnings
import streamlit as st
from dotenv import load_dotenv
from utils.data_loader import load_dataset
from utils.model_trainer import train_all_models
from sklearn.exceptions import ConvergenceWarning

load_dotenv()
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Insurance Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS injection ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Global font & background */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #0a0e1a !important;
        border-right: 1px solid #1e2535;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #c8d0e8 !important;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] > label {
        border: 1px solid transparent;
        border-radius: 8px !important;
        margin: 2px 0 !important;
        padding: 6px 8px !important;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
        background: #151e33 !important;
        border-color: #2a3657;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] > label[data-checked="true"] {
        background: #1b2b4f !important;
        border-color: #335ea8;
    }
    section[data-testid="stSidebar"] .stFileUploader label {
        color: #d7e3ff !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] .stFileUploader section {
        border: 1px dashed #35507f !important;
        border-radius: 10px !important;
        background: #0f172b !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f7f8fc;
        border: 1px solid #e8eaf0;
        border-radius: 12px;
        padding: 18px 20px;
    }

    /* Dataframe */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Buttons */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: all 0.15s;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 20px 0 24px 0;">
            <div style="font-size: 20px; font-weight: 600; color: #e8eaf0; letter-spacing: -0.3px;">
                🧬 InsureML
            </div>
            <div style="font-size: 12px; color: #5a6480; margin-top: 4px; font-family: 'DM Mono', monospace;">
                v1.0 · Insurance Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        options=[
            "01 · Business Case",
            "02 · Data Exploration",
            "03 · Model Comparison",
            "04 · Explainability",
            "05 · Hyperparameter Tuning",
        ],
        label_visibility="collapsed",
        key="main_nav_radio",
    )

    st.markdown(
        "<div style='margin-top: 32px; border-top: 1px solid #1e2535; padding-top: 20px;'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="font-size: 11px; color: #3a4060; margin-top: 24px; line-height: 1.6;">
        Dataset: US Medical Insurance Costs<br>
        Source: Kaggle / R Datasets<br>
        1,338 rows · 7 features
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Data & model loading ─────────────────────────────────────────────────────
df = load_dataset()
try:
    training = train_all_models(df)
except Exception:
    st.error("Required model engines unavailable.")
    st.stop()

# ── Page routing ─────────────────────────────────────────────────────────────
if page == "01 · Business Case":
    from pages.p01_business import render
    render(df)

elif page == "02 · Data Exploration":
    from pages.p02_data_viz import render
    render(df)

elif page == "03 · Model Comparison":
    from pages.p03_prediction import render
    render(df, training)

elif page == "04 · Explainability":
    from pages.p04_explainability import render
    render(df, training)

elif page == "05 · Hyperparameter Tuning":
    from pages.p05_tuning import render
    render(df, training)
