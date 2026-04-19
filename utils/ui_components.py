"""
utils/ui_components.py
──────────────────────
Reusable styled components shared across pages.
"""

import streamlit as st


def page_header(title: str, subtitle: str, emoji: str = ""):
    """Render a consistent dark hero header for each page."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0a0e1a 0%, #0f1e3d 60%, #0e2b52 100%);
            padding: 32px 36px;
            border-radius: 16px;
            margin-bottom: 28px;
            border: 1px solid #1a2540;
        ">
            <div style="font-size: 11px; font-family: 'DM Mono', monospace; color: #4a6090;
                        letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px;">
                InsureML · Intelligence Dashboard
            </div>
            <h1 style="margin: 0; font-size: 28px; font-weight: 600; color: #e8edf8;
                       letter-spacing: -0.5px; line-height: 1.2;">
                {emoji} {title}
            </h1>
            <p style="margin: 8px 0 0; font-size: 15px; color: #6a7fa8; font-weight: 400;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str = "", color: str = "#1a2e58"):
    """Render a styled metric card."""
    delta_html = (
        f'<div style="font-size: 12px; color: #6db87a; margin-top: 4px;">{delta}</div>'
        if delta else ""
    )
    st.markdown(
        f"""
        <div style="
            background: #111a2b;
            border: 1px solid #24314d;
            border-left: 4px solid {color};
            border-radius: 10px;
            padding: 16px 20px;
        ">
            <div style="font-size: 12px; color: #93a6cc; font-weight: 500;
                        text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;">
                {label}
            </div>
            <div style="font-size: 24px; font-weight: 600; color: #e8edf8;">
                {value}
            </div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str):
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px; margin: 32px 0 16px;">
            <div style="width: 3px; height: 20px; background: #2563eb; border-radius: 2px;"></div>
            <h3 style="margin: 0; font-size: 17px; font-weight: 600; color: #e6edff;">{text}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_box(text: str, icon: str = "💡"):
    st.markdown(
        f"""
        <div style="
            background: #111a2b;
            border: 1px solid #27395d;
            border-radius: 10px;
            padding: 14px 18px;
            margin: 16px 0;
            font-size: 14px;
            color: #c9d7f5;
            line-height: 1.6;
        ">
            <strong style="color: #7ab0ff;">{icon} Insight: </strong>{text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def tech_decision_box(decision: str, reason: str):
    """Show a why-we-chose-this technical decision box."""
    st.markdown(
        f"""
        <div style="
            background: #111a2b;
            border: 1px solid #27395d;
            border-radius: 10px;
            padding: 14px 18px;
            margin: 8px 0;
            font-size: 13.5px;
        ">
            <div style="font-weight: 600; color: #e8edff; margin-bottom: 5px;">⚙️ {decision}</div>
            <div style="color: #b8c7e8; line-height: 1.6;">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
