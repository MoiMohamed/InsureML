"""
pages/p02_data_viz.py
──────────────────────
Page 2: Data Exploration & Visual Insights

PURPOSE:
  Show the key patterns in the data that motivate the model choices.
  Every chart should answer a business question, not just display numbers.

CHART CHOICES (with rationale):
  - Histogram of charges: shows bimodal distribution (smokers vs non-smokers)
    — motivates non-linear models
  - Box plots by smoker: quantifies the smoking effect unambiguously
  - Scatter BMI vs charges (colored by smoker): reveals the interaction term
    — the most important visual in the whole project
  - Correlation heatmap: shows age and BMI are the numeric predictors
  - Region bar chart: context, shows variation is modest vs smoker effect
  - Scenario simulator: actionable pricing tool for business stakeholders
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency
from utils.ui_components import insight_box, page_header, section_title

TARGET_COL = "charges"

# Consistent color palette
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"
AMBER = "#d97706"
PURPLE = "#7c3aed"
GRAY = "#6b7280"
LIGHT_BG = "#f8faff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": LIGHT_BG,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.labelcolor": "#374151",
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
    "axes.titlesize": 13,
    "axes.titleweight": "600",
    "axes.titlecolor": "#1a2035",
    "axes.labelsize": 11,
})


def render(df: pd.DataFrame):
    page_header(
        title="Data Exploration",
        subtitle="Visual evidence for model design choices",
        emoji="📊",
    )

    tab1, tab2, tab3 = st.tabs([
        "📈 Distributions",
        "🔀 Feature Interactions",
        "💸 Pricing Simulator",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        section_title("Charge Distribution (Bimodal Pattern)")
        insight_box(
            "Notice the two peaks in the histogram — the right hump is almost entirely smokers. "
            "This bimodal distribution is why a single linear model struggles: smokers and "
            "non-smokers follow different pricing regimes."
        )
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            non_smoker_charges = df[df["smoker"] == "no"][TARGET_COL]
            smoker_charges = df[df["smoker"] == "yes"][TARGET_COL]
            bins = np.linspace(df[TARGET_COL].min(), df[TARGET_COL].max(), 40)
            ax.hist(non_smoker_charges, bins=bins, color=BLUE, alpha=0.7, label="Non-smoker")
            ax.hist(smoker_charges, bins=bins, color=RED, alpha=0.75, label="Smoker")
            ax.set_xlabel("Annual Charge ($)")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Insurance Charges")
            ax.legend(fontsize=10)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            st.pyplot(fig, width="stretch")

        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            data = [
                df[df["smoker"] == "no"][TARGET_COL].values,
                df[df["smoker"] == "yes"][TARGET_COL].values,
            ]
            bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                            medianprops={"color": "white", "linewidth": 2})
            bp["boxes"][0].set_facecolor(BLUE)
            bp["boxes"][1].set_facecolor(RED)
            for box in bp["boxes"]:
                box.set_alpha(0.85)
            ax.set_xticklabels(["Non-smoker", "Smoker"])
            ax.set_ylabel("Annual Charge ($)")
            ax.set_title("Charge Spread by Smoking Status")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))
            st.pyplot(fig, width="stretch")

        section_title("Age & BMI Distributions")
        c3, c4 = st.columns(2)
        with c3:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(df["age"], bins=30, color=PURPLE, alpha=0.8, edgecolor="white")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Count")
            ax.set_title("Age Distribution")
            st.pyplot(fig, width="stretch")

        with c4:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(df["bmi"], bins=30, color=GREEN, alpha=0.8, edgecolor="white")
            ax.axvline(30, color=RED, linestyle="--", linewidth=1.5, label="BMI 30 (obese)")
            ax.set_xlabel("BMI")
            ax.set_ylabel("Count")
            ax.set_title("BMI Distribution")
            ax.legend(fontsize=10)
            st.pyplot(fig, width="stretch")

        section_title("Correlation Heatmap (Numeric Features)")
        num_df = df.select_dtypes(include=np.number)
        corr = num_df.corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=10)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=10)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(corr.iloc[i, j]) > 0.4 else "#374151")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig, width="stretch")
        insight_box(
            "Among numeric features, age (r=0.30) and BMI (r=0.20) have the strongest "
            "correlation with charges. The weak linear correlations confirm we need "
            "non-linear models — the real signal comes from feature interactions."
        )

    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        section_title("The Key Interaction: BMI × Smoking Status")
        insight_box(
            "This is the single most important chart in the project. Notice how high BMI "
            "only drives costs up for smokers — non-smokers show no clear BMI effect. "
            "This interaction term is invisible to linear models but captured perfectly "
            "by tree-based methods. This justifies using XGBoost/Random Forest."
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        nonsmokers = df[df["smoker"] == "no"]
        smokers = df[df["smoker"] == "yes"]
        ax.scatter(nonsmokers["bmi"], nonsmokers[TARGET_COL],
                   color=BLUE, alpha=0.4, s=18, label="Non-smoker", zorder=2)
        ax.scatter(smokers["bmi"], smokers[TARGET_COL],
                   color=RED, alpha=0.5, s=22, label="Smoker", zorder=3)
        ax.axvline(30, color=GRAY, linestyle="--", linewidth=1, alpha=0.7, label="BMI 30 threshold")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Annual Charge ($)")
        ax.set_title("BMI vs Insurance Charges — colored by Smoking Status")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))
        ax.legend(fontsize=10)
        st.pyplot(fig, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            section_title("Age vs Charges")
            fig, ax = plt.subplots(figsize=(5.5, 4))
            for smoker_status, color in [("no", BLUE), ("yes", RED)]:
                sub = df[df["smoker"] == smoker_status]
                ax.scatter(sub["age"], sub[TARGET_COL], color=color,
                           alpha=0.35, s=14, label=f"{'Non-s' if smoker_status=='no' else 'S'}moker")
                z = np.polyfit(sub["age"], sub[TARGET_COL], 1)
                p = np.poly1d(z)
                x_line = np.linspace(sub["age"].min(), sub["age"].max(), 100)
                ax.plot(x_line, p(x_line), color=color, linewidth=1.8, alpha=0.9)
            ax.set_xlabel("Age")
            ax.set_ylabel("Charge ($)")
            ax.set_title("Age vs Charges")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))
            ax.legend(fontsize=10)
            st.pyplot(fig, width="stretch")

        with c2:
            section_title("Average Charges by Region")
            region_stats = (
                df.groupby("region")[TARGET_COL].mean().sort_values(ascending=True)
            )
            fig, ax = plt.subplots(figsize=(5.5, 4))
            bars = ax.barh(region_stats.index, region_stats.values, color=BLUE,
                           alpha=0.85, height=0.55)
            for bar, val in zip(bars, region_stats.values):
                ax.text(val + 80, bar.get_y() + bar.get_height() / 2,
                        f"${val:,.0f}", va="center", fontsize=10, color="#374151")
            ax.set_xlabel("Average Annual Charge ($)")
            ax.set_title("Regional Charge Differences")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            st.pyplot(fig, width="stretch")

        section_title("Charges by Number of Dependents")
        child_stats = df.groupby("children")[TARGET_COL].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(child_stats["children"].astype(str), child_stats["mean"],
               yerr=child_stats["std"] / 2, capsize=5,
               color=AMBER, alpha=0.85, width=0.5, ecolor="#6b7280")
        ax.set_xlabel("Number of Dependents")
        ax.set_ylabel("Average Charge ($)")
        ax.set_title("Mean Charges by Number of Children (± ½ Std Dev)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))
        st.pyplot(fig, width="stretch")

        section_title("Categorical Association with Charges (Cramer's V)")
        st.markdown(
            "<p style='color:#5a6480; font-size:14px;'>"
            "Cramer's V measures association strength between categorical variables. "
            "Since charges are numeric, we bin charges into quintiles before computing "
            "association with each categorical feature.</p>",
            unsafe_allow_html=True,
        )

        def cramers_v(x: pd.Series, y: pd.Series) -> float:
            table = pd.crosstab(x, y)
            if table.shape[0] < 2 or table.shape[1] < 2:
                return 0.0
            chi2 = chi2_contingency(table, correction=False)[0]
            n = table.to_numpy().sum()
            if n == 0:
                return 0.0
            phi2 = chi2 / n
            r, k = table.shape
            # Bias-corrected Cramer's V.
            phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
            r_corr = r - ((r - 1) ** 2) / max(n - 1, 1)
            k_corr = k - ((k - 1) ** 2) / max(n - 1, 1)
            denom = min((k_corr - 1), (r_corr - 1))
            if denom <= 0:
                return 0.0
            return float(np.sqrt(phi2_corr / denom))

        cat_cols = [c for c in df.columns if df[c].dtype == "object"]
        charge_bins = pd.qcut(df[TARGET_COL], q=5, duplicates="drop")
        cv_rows = []
        for col in cat_cols:
            cv_rows.append(
                {"Categorical Feature": col, "Cramer's V vs charges_bin": cramers_v(df[col], charge_bins)}
            )
        cv_df = pd.DataFrame(cv_rows).sort_values("Cramer's V vs charges_bin", ascending=False)
        st.dataframe(
            cv_df.style.format({"Cramer's V vs charges_bin": "{:.3f}"}),
            width="stretch",
            hide_index=True,
        )
        insight_box(
            "Higher Cramer's V means stronger categorical association with charge level buckets. "
            "In this dataset, smoker typically dominates."
        )

    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        section_title("Policy Pricing Scenario Simulator")
        st.markdown(
            "<p style='color:#5a6480; font-size:14px; margin-bottom:20px;'>"
            "Adjust actuarial assumptions to project how portfolio changes affect average premiums. "
            "This is the 'what-if' tool for business stakeholders.</p>",
            unsafe_allow_html=True,
        )
        baseline_avg = float(df[TARGET_COL].mean())
        base_smoker_rate = float((df["smoker"] == "yes").mean())

        c1, c2 = st.columns([1.2, 1])
        with c1:
            age_factor = st.slider("Age-adjustment factor", 0.8, 1.3, 1.0, 0.01,
                                   help="1.0 = current age distribution unchanged")
            bmi_factor = st.slider("BMI-adjustment factor", 0.8, 1.3, 1.0, 0.01,
                                   help="1.0 = current BMI distribution unchanged")
            smoker_risk = st.slider("Smoker risk multiplier", 1.0, 2.0, 1.0, 0.01,
                                    help="How much more expensive smokers are vs baseline")
            projected_smoker_rate = st.slider(
                "Projected smoker portfolio share", 0.0, 0.6, base_smoker_rate, 0.01,
                help=f"Current rate: {base_smoker_rate:.1%}"
            )

        policy_score = age_factor * bmi_factor * (
            (1 - projected_smoker_rate) + projected_smoker_rate * smoker_risk
        )
        projected_avg = baseline_avg * policy_score
        delta = projected_avg - baseline_avg

        with c2:
            st.markdown(
                f"""
                <div style="background: #f8faff; border: 1px solid #dbe4f8;
                            border-radius: 14px; padding: 28px 24px; margin-top: 12px;">
                    <div style="font-size: 12px; font-weight: 600; color: #6b7db8;
                                text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 16px;">
                        Simulation Results
                    </div>
                    <div style="font-size: 13px; color: #8890a8; margin-bottom: 6px;">
                        Baseline avg charge
                    </div>
                    <div style="font-size: 26px; font-weight: 600; color: #1a2035; margin-bottom: 16px;">
                        ${baseline_avg:,.2f}
                    </div>
                    <div style="font-size: 13px; color: #8890a8; margin-bottom: 6px;">
                        Projected avg charge
                    </div>
                    <div style="font-size: 26px; font-weight: 600; color:
                        {'#dc2626' if projected_avg > baseline_avg else '#059669'};
                        margin-bottom: 16px;">
                        ${projected_avg:,.2f}
                    </div>
                    <div style="border-top: 1px solid #dbe4f8; padding-top: 16px;">
                        <div style="font-size: 13px; color: #8890a8; margin-bottom: 6px;">
                            Delta per policy
                        </div>
                        <div style="font-size: 22px; font-weight: 600; color:
                            {'#dc2626' if delta > 0 else '#059669'};">
                            {'▲' if delta > 0 else '▼'} ${abs(delta):,.2f}
                            ({'increased' if delta > 0 else 'reduced'} risk)
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Progress bar visualization
        ratio = projected_avg / baseline_avg
        bar_pct = min(ratio * 50, 100)
        bar_color = "#dc2626" if ratio > 1.05 else ("#059669" if ratio < 0.95 else "#2563eb")
        st.markdown(
            f"""
            <div style="margin: 16px 0 4px; font-size: 12px; color: #8890a8;">
                Projected charge as % of baseline
            </div>
            <div style="background: #e8eaf0; border-radius: 6px; height: 12px; overflow: hidden;">
                <div style="width: {bar_pct}%; background: {bar_color};
                            height: 100%; border-radius: 6px; transition: width 0.3s;"></div>
            </div>
            <div style="font-size: 12px; color: {bar_color}; margin-top: 4px; font-weight: 600;">
                {ratio:.2%} of baseline
            </div>
            """,
            unsafe_allow_html=True,
        )
