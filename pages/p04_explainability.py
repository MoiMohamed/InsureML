"""
pages/p04_explainability.py
────────────────────────────
Page 4: Feature Importance & SHAP Explainability

TECHNICAL DECISIONS:
  - WHY SHAP?
    SHAP (SHapley Additive exPlanations) is game-theory-grounded: each feature
    gets the exact credit for its contribution to a prediction. Unlike
    permutation importance, it's additive — values sum to the model output.
    This makes it the gold standard for explaining individual predictions.

  - WHY Permutation Importance too?
    SHAP can fail for non-tree models (falls back to slow KernelExplainer).
    Permutation importance works for ANY sklearn estimator. It's also
    model-agnostic and measures real impact: "how much worse does the model
    get when I shuffle this feature?"

  - WHY both?
    Permutation importance = global feature ranking (which features matter most
    across the dataset). SHAP beeswarm = shows direction + distribution
    (does high BMI increase or decrease charges? by how much?).

  - SHAP Explainer selection logic:
    TreeExplainer: fast, exact, works for RF/GBM/XGBoost/LightGBM
    LinearExplainer: fast, exact, works for Linear/Ridge/Lasso
    KernelExplainer: slow, approximate, works for anything (fallback)
    We auto-detect which to use based on the model class.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.inspection import permutation_importance
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from utils.ui_components import insight_box, page_header, section_title, tech_decision_box

TARGET_COL = "charges"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "#f8faff",
    "figure.facecolor": "white",
    "axes.labelcolor": "#0f172a",
    "axes.titlecolor": "#0f172a",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def _clean_feature_name(name: str) -> str:
    if name.startswith("num__"):
        return name.replace("num__", "")
    if name.startswith("cat__"):
        return name.replace("cat__", "")
    return name


def _get_shap_explainer(model, X_sample):
    """Auto-select the most appropriate SHAP explainer for the given model."""
    try:
        from xgboost import XGBRegressor
        xgb_available = True
    except Exception:
        xgb_available = False

    try:
        from lightgbm import LGBMRegressor
        lgbm_available = True
    except Exception:
        lgbm_available = False

    tree_types = [RandomForestRegressor, GradientBoostingRegressor]
    if xgb_available:
        tree_types.append(XGBRegressor)
    if lgbm_available:
        tree_types.append(LGBMRegressor)

    if isinstance(model, tuple(tree_types)):
        return shap.TreeExplainer(model)

    if isinstance(model, (LinearRegression, Ridge, Lasso)):
        return shap.LinearExplainer(model, X_sample)

    # Universal fallback (slow on large datasets)
    return shap.KernelExplainer(model.predict, shap.kmeans(X_sample, 30))


def render(df: pd.DataFrame, training: dict):
    page_header(
        title="Explainability",
        subtitle="SHAP values + permutation importance — understand every prediction",
        emoji="🔍",
    )

    X_test: pd.DataFrame = training["X_test"].copy()
    y_test: pd.Series = training["y_test"].copy()
    model_map: dict = training["model_map"]
    leaderboard: pd.DataFrame = training["leaderboard"]

    explainable_models = [name for name in model_map if "Ensemble" not in name]
    if not explainable_models:
        st.error("No explainable single-model pipelines available.")
        return

    leaderboard_order = [
        name for name in leaderboard["Model"].tolist() if name in explainable_models
    ]
    default_model = (
        leaderboard_order[0] if leaderboard_order else explainable_models[0]
    )

    model_name = st.selectbox(
        "Model to explain",
        options=leaderboard_order if leaderboard_order else explainable_models,
        index=0,
        help="Ensembles are excluded here to ensure faithful SHAP explanations.",
    )
    best_pipe = model_map[model_name]

    if model_name != default_model:
        st.info(f"Explaining: **{model_name}**")
    else:
        st.info(
            f"Explaining: **{model_name}** (best available single model on leaderboard)"
        )
    st.markdown(
        "- **Global importance**: which features matter most overall\n"
        "- **Direction**: whether a feature pushes price up or down\n"
        "- **Local explanation**: why one specific prediction is high/low"
    )

    preprocessor = best_pipe.named_steps["preprocessor"]
    model = best_pipe.named_steps["model"]

    feature_names = list(preprocessor.get_feature_names_out())
    X_transformed = preprocessor.transform(X_test)
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    # ── Permutation importance ────────────────────────────────────────────────
    section_title("Permutation Feature Importance")
    tech_decision_box(
        "What is permutation importance?",
        "For each feature, we randomly shuffle its values across all test samples and "
        "measure how much RMSE increases. If shuffling a feature makes predictions much "
        "worse, that feature was important. If RMSE barely changes, the feature can be "
        "removed without affecting model quality. Works for ANY sklearn pipeline.",
    )

    with st.spinner("Computing permutation importance (10 repeats)…"):
        perm = permutation_importance(
            best_pipe,
            X_test,
            y_test,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            n_repeats=10,
        )

    perm_df = (
        pd.DataFrame({
            "Feature": X_test.columns,
            "Mean Importance (RMSE increase)": perm.importances_mean,
            "Std Dev": perm.importances_std,
        })
        .sort_values("Mean Importance (RMSE increase)", ascending=True)
        .tail(10)
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(
        perm_df["Feature"],
        perm_df["Mean Importance (RMSE increase)"],
        xerr=perm_df["Std Dev"],
        capsize=3,
        color="#2563eb",
        alpha=0.85,
        height=0.6,
        ecolor="#6b7280",
    )
    ax.set_xlabel("Mean RMSE increase when feature is shuffled")
    ax.set_title(f"Top 10 Features by Permutation Importance — {model_name}")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{width:.1f}", va="center", fontsize=9, color="#374151")
    st.pyplot(fig, width="stretch")

    insight_box(
        "Smoker status dominates feature importance — shuffling it causes the largest "
        "RMSE spike. This confirms our earlier EDA finding: smoking status is the "
        "single biggest cost driver, followed by age and BMI."
    )

    # ── SHAP ─────────────────────────────────────────────────────────────────
    section_title("SHAP Value Analysis")
    tech_decision_box(
        "Why SHAP over other explainability methods?",
        "SHAP is the only method that is simultaneously: (1) consistent — if a model relies "
        "more on a feature, its SHAP value increases; (2) locally accurate — SHAP values "
        "sum to the exact model output minus the baseline; (3) missingness-preserving. "
        "LIME and feature weights don't have all three properties.",
    )

    if not SHAP_AVAILABLE:
        st.error("SHAP unavailable.")
        return

    sample_n = min(200, len(X_transformed_df))
    X_sample = X_transformed_df.iloc[:sample_n]

    with st.spinner("Computing SHAP values…"):
        try:
            explainer = _get_shap_explainer(model, X_sample)
            shap_values = explainer(X_sample)

            mean_abs = np.abs(shap_values.values).mean(axis=0)
            shap_rank = (
                pd.DataFrame(
                    {
                        "Feature": [_clean_feature_name(f) for f in feature_names],
                        "Mean |SHAP|": mean_abs,
                    }
                )
                .sort_values("Mean |SHAP|", ascending=False)
                .head(12)
            )
            section_title("Most Influential Features (Mean |SHAP|)")
            st.dataframe(
                shap_rank.style.format({"Mean |SHAP|": "{:.3f}"}),
                width="stretch",
                hide_index=True,
            )
            top_row = shap_rank.iloc[0]
            st.caption(
                f"Highest overall influence: `{top_row['Feature']}` "
                f"(mean |SHAP| = {top_row['Mean |SHAP|']:.3f})."
            )

            # ── Beeswarm plot ─────────────────────────────────────────────────
            st.markdown(
                "<p style='font-size:14px; color:#5a6480; margin-bottom: 8px;'>"
                "<strong>Beeswarm plot</strong>: each dot is one test sample. "
                "X-axis = SHAP value (positive = pushes prediction up). "
                "Color = feature value (red = high, blue = low).</p>",
                unsafe_allow_html=True,
            )
            plt.figure(figsize=(10, 5))
            shap.plots.beeswarm(shap_values, show=False, max_display=12, plot_size=None)
            st.pyplot(plt.gcf(), clear_figure=True, width="stretch")

            # ── Waterfall for one prediction ───────────────────────────────────
            section_title("Single Prediction Explanation (Waterfall)")
            st.markdown(
                "<p style='font-size:14px; color:#5a6480;'>"
                "Shows how each feature nudges the prediction above or below the baseline "
                "for the first test sample. This is what you'd show a customer explaining "
                "why their premium is what it is.</p>",
                unsafe_allow_html=True,
            )
            plt.figure(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False, max_display=10)
            st.pyplot(plt.gcf(), clear_figure=True, width="stretch")

            # ── Feature dependence for top feature ────────────────────────────
            if len(shap_values.values.shape) > 1:
                top_feat_idx = np.abs(shap_values.values).mean(axis=0).argmax()
                top_feat = _clean_feature_name(feature_names[top_feat_idx])
                section_title(f"Dependence Plot — {top_feat}")
                st.markdown(
                    f"<p style='font-size:14px; color:#5a6480;'>"
                    f"How does <strong>{top_feat}</strong>'s SHAP value change with its raw value? "
                    "Color shows the interaction with the most correlated other feature.</p>",
                    unsafe_allow_html=True,
                )
                plt.figure(figsize=(8, 4.5))
                shap.plots.scatter(shap_values[:, top_feat_idx], show=False)
                st.pyplot(plt.gcf(), clear_figure=True, width="stretch")

        except Exception:
            st.error("SHAP computation unavailable for selected model.")

    # ── Technical summary ─────────────────────────────────────────────────────
    section_title("Explainability Methods — Summary")
    comp_data = pd.DataFrame([
        {"Method": "Permutation Importance", "Scope": "Global",
         "Model-agnostic": "✅", "Directional": "❌", "Additive": "❌",
         "Speed": "Fast"},
        {"Method": "SHAP Beeswarm", "Scope": "Global",
         "Model-agnostic": "✅", "Directional": "✅", "Additive": "✅",
         "Speed": "Medium"},
        {"Method": "SHAP Waterfall", "Scope": "Local (per prediction)",
         "Model-agnostic": "✅", "Directional": "✅", "Additive": "✅",
         "Speed": "Medium"},
        {"Method": "Coefficient weights", "Scope": "Global",
         "Model-agnostic": "❌ (linear only)", "Directional": "✅", "Additive": "✅",
         "Speed": "Instant"},
    ])
    st.dataframe(comp_data, width="stretch", hide_index=True)
