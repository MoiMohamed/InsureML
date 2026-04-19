"""
pages/p01_business.py
──────────────────────
Page 1: Business Case + Dataset Overview

PURPOSE:
  Sets the commercial context. Insurance companies lose money on mispriced policies.
  This page frames the ML solution as a business tool — not just an academic exercise.
"""

import pandas as pd
import streamlit as st
from utils.ui_components import insight_box, metric_card, page_header, section_title

TARGET_COL = "charges"


def render(df: pd.DataFrame):
    page_header(
        title="Business Case & Dataset",
        subtitle="Why insurance pricing is broken — and how ML fixes it",
        emoji="💼",
    )

    # ── Problem statement ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 28px;
        ">
            <div style="background: linear-gradient(135deg, #fff7ed, #fef3c7); border: 1px solid #fcd34d;
                        border-radius: 12px; padding: 20px 22px;">
                <div style="font-size: 13px; font-weight: 600; color: #92400e; text-transform: uppercase;
                             letter-spacing: 0.8px; margin-bottom: 8px;">The Problem</div>
                <div style="font-size: 14px; color: #78350f; line-height: 1.7;">
                    Actuaries use rigid demographic brackets. A 35-year-old non-smoker with BMI 32
                    gets the same quote as one with BMI 21 — despite 3× cost difference.
                </div>
            </div>
            <div style="background: linear-gradient(135deg, #eff6ff, #dbeafe); border: 1px solid #93c5fd;
                        border-radius: 12px; padding: 20px 22px;">
                <div style="font-size: 13px; font-weight: 600; color: #1e40af; text-transform: uppercase;
                             letter-spacing: 0.8px; margin-bottom: 8px;">The Opportunity</div>
                <div style="font-size: 14px; color: #1e3a8a; line-height: 1.7;">
                    Granular ML models predict individual charges with ±$2k accuracy,
                    enabling risk-based pricing that's fairer to low-risk customers
                    and profitable for insurers.
                </div>
            </div>
            <div style="background: linear-gradient(135deg, #f0fdf4, #dcfce7); border: 1px solid #86efac;
                        border-radius: 12px; padding: 20px 22px;">
                <div style="font-size: 13px; font-weight: 600; color: #166534; text-transform: uppercase;
                             letter-spacing: 0.8px; margin-bottom: 8px;">Our Solution</div>
                <div style="font-size: 14px; color: #14532d; line-height: 1.7;">
                    Benchmark 7 ML models + an ensemble. Explain every prediction.
                    Give actuaries a live tool to simulate premium scenarios
                    before policy changes.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Dataset metrics ──────────────────────────────────────────────────────
    section_title("Dataset at a Glance")
    avg_charge = df[TARGET_COL].mean()
    smoker_pct = (df["smoker"] == "yes").mean() * 100
    smoker_avg = df[df["smoker"] == "yes"][TARGET_COL].mean()
    non_smoker_avg = df[df["smoker"] == "no"][TARGET_COL].mean()
    smoker_multiple = smoker_avg / non_smoker_avg

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total records", f"{df.shape[0]:,}", color="#2563eb")
    with c2:
        metric_card("Avg annual charge", f"${avg_charge:,.0f}", color="#0891b2")
    with c3:
        metric_card("Smoker share", f"{smoker_pct:.1f}%", color="#dc2626")
    with c4:
        metric_card(
            "Smoker cost multiplier",
            f"{smoker_multiple:.1f}×",
            "Smokers cost {:.1f}× more".format(smoker_multiple),
            color="#d97706",
        )

    # ── Data preview ─────────────────────────────────────────────────────────
    section_title("Dataset Preview")
    st.dataframe(df.head(10), width="stretch", hide_index=True)

    # ── Data dictionary ───────────────────────────────────────────────────────
    section_title("Data Dictionary")
    DESCRIPTIONS = {
        "age": ("Continuous", "Age of the primary beneficiary in years (18–64)"),
        "sex": ("Categorical", "Biological sex of the beneficiary: male / female"),
        "bmi": ("Continuous", "Body Mass Index (kg/m²). Threshold: 30 = obese"),
        "children": ("Discrete", "Number of dependents covered under the plan (0–5)"),
        "smoker": ("Binary", "Whether the beneficiary is a current smoker: yes / no"),
        "region": ("Categorical", "US region: northeast, northwest, southeast, southwest"),
        "charges": ("Target (USD)", "Actual medical insurance billed — what we predict"),
    }
    dict_rows = []
    for col in df.columns:
        dtype, desc = DESCRIPTIONS.get(col, ("Unknown", "—"))
        dict_rows.append(
            {"Feature": col, "Type": dtype, "Description": desc,
             "Null count": int(df[col].isna().sum()),
             "Unique values": int(df[col].nunique())}
        )
    st.dataframe(pd.DataFrame(dict_rows), width="stretch", hide_index=True)

    # ── Summary statistics ────────────────────────────────────────────────────
    section_title("Summary Statistics")
    describe_df = df.describe(include="all").transpose().reset_index().rename(columns={"index": "feature"})
    st.dataframe(describe_df, width="stretch", hide_index=True)

    insight_box(
        "Smoking status is the single largest cost driver — smokers are charged "
        f"{smoker_multiple:.1f}× more on average (${smoker_avg:,.0f} vs ${non_smoker_avg:,.0f}). "
        "Any model that ignores smoker status will have high systematic error.",
    )
