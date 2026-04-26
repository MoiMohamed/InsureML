"""
pages/p03_prediction.py
────────────────────────
Page 3: Model Comparison + Live Prediction

PURPOSE:
  Let users compare model performance and make live predictions.

DESIGN DECISIONS:
  - Leaderboard first: business stakeholders need to know which model to trust.
  - Model descriptions: every model has a plain-English explanation of why it's
    included and what its trade-offs are.
  - Prediction form: sliders match the actual data ranges so predictions are
    always within the training distribution.
  - Ensemble option: shows how combining models beats any single model.
  - Actual vs Predicted scatter: the best diagnostic for regression model quality.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.ui_components import insight_box, page_header, section_title, tech_decision_box
from utils.model_trainer import train_all_models

TARGET_COL = "charges"

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "#111a2b",
    "figure.facecolor": "#0b1220",
    "axes.labelcolor": "#dbe7ff",
    "axes.titlecolor": "#e8edff",
    "xtick.color": "#b8c7e8",
    "ytick.color": "#b8c7e8",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def _collect_user_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Render input widgets for each feature and return a single-row DataFrame."""
    x_cols = [c for c in df.columns if c != TARGET_COL]
    inputs = {}
    for col in x_cols:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            options = sorted(df[col].dropna().unique().tolist())
            inputs[col] = st.selectbox(f"**{col.upper()}**", options)
        elif np.issubdtype(df[col].dtype, np.integer):
            lo, hi = int(df[col].min()), int(df[col].max())
            inputs[col] = st.slider(f"**{col.upper()}**", lo, hi, int(df[col].median()))
        else:
            lo, hi = float(df[col].min()), float(df[col].max())
            inputs[col] = st.slider(f"**{col.upper()}**", lo, hi, float(df[col].median()), 0.1)
    return pd.DataFrame([inputs])


def render(df: pd.DataFrame, training: dict):
    page_header(
        title="Model Comparison & Prediction",
        subtitle="Benchmark 7 models + ensemble · Select and predict live",
        emoji="🤖",
    )

    # ── Technical architecture decisions (at top) ────────────────────────────
    section_title("Technical Architecture Decisions")
    tech_decision_box(
        "Why StandardScaler?",
        "Linear models and neural networks are sensitive to feature scale. BMI ranges "
        "15–55 while children ranges 0–5. Without scaling, the optimizer treats BMI "
        "gradients as much larger. StandardScaler normalises to μ=0, σ=1 per feature, "
        "applied only to train data to prevent leakage.",
    )
    tech_decision_box(
        "Why OneHotEncoder for categoricals?",
        "sex, smoker, and region are nominal (no natural order). Label encoding would "
        "imply 'southeast > northwest' which is false. OHE creates binary indicator "
        "columns — 6 total for 3 categorical features.",
    )
    tech_decision_box(
        "Why an 80/20 train-test split?",
        "With 1,338 rows, 80% (~1,070) gives enough training signal for boosted trees "
        "(which need diversity). 20% (~268) is enough to estimate RMSE within ±5% "
        "confidence. A fixed random_state=42 ensures reproducibility.",
    )
    tech_decision_box(
        "Why Pipeline over manual preprocessing?",
        "A Pipeline chains preprocessor + model so that .predict(new_df) handles both "
        "transform and inference in one call. More importantly, it prevents data "
        "leakage: the scaler never sees test data during fitting.",
    )
    insight_box(
        "Smoking status is the single largest cost driver — smokers are charged 3.8× "
        "more on average ($32,050 vs $8,434). Any model that ignores smoker status "
        "will have high systematic error."
    )

    # ── Feature include/exclude selector ─────────────────────────────────────
    section_title("Feature Selection (Include / Exclude)")
    all_features = [c for c in df.columns if c != TARGET_COL]
    selected_features = st.multiselect(
        "Select features to include in training",
        options=all_features,
        default=all_features,
        key="model_cmp_feature_selector",
    )
    if not selected_features:
        st.error("Select at least one feature.")
        return

    # Re-train benchmark using selected features (cached by feature tuple).
    training = train_all_models(df, selected_features=tuple(selected_features))

    leaderboard = training["leaderboard"].copy()
    baseline_metrics_map = training.get("baseline_metrics_map", {})
    search_trials_map = training.get("search_trials_map", {})

    # Style helper for the single live leaderboard table.
    def highlight_best(row):
        if row.name == 0:
            return ["background: #eff6ff; font-weight: 600; color: #1e3a8a"] * len(row)
        return [""] * len(row)

    section_title("Tuning Improvement (Baseline vs Best Random Search)")
    improvement_rows = []
    for r in training["results"]:
        if "Ensemble" in r.name:
            continue
        baseline_rmse = baseline_metrics_map.get(r.name, {}).get("rmse", np.nan)
        if np.isfinite(baseline_rmse) and baseline_rmse > 0:
            gain_pct = 100.0 * (baseline_rmse - r.rmse) / baseline_rmse
        else:
            gain_pct = np.nan
        improvement_rows.append(
            {
                "Model": r.name,
                "Baseline RMSE": baseline_rmse,
                "Best RMSE": r.rmse,
                "RMSE Gain %": gain_pct,
            }
        )
    improve_df = pd.DataFrame(improvement_rows).sort_values("Best RMSE")
    if not improve_df.empty:
        fig_imp, ax_imp = plt.subplots(figsize=(8.5, 4.8))
        y = np.arange(len(improve_df))
        ax_imp.barh(y - 0.2, improve_df["Baseline RMSE"], height=0.35, label="Baseline", alpha=0.65)
        ax_imp.barh(y + 0.2, improve_df["Best RMSE"], height=0.35, label="Best (random search)", alpha=0.9)
        ax_imp.set_yticks(y)
        ax_imp.set_yticklabels(improve_df["Model"])
        ax_imp.set_xlabel("RMSE on hold-out test split")
        ax_imp.set_title("How much tuning improved each model")
        ax_imp.legend()
        st.pyplot(fig_imp, width="stretch")
        st.dataframe(
            improve_df.style.format(
                {"Baseline RMSE": "${:,.0f}", "Best RMSE": "${:,.0f}", "RMSE Gain %": "{:+.2f}%"}
            ),
            width="stretch",
            hide_index=True,
        )

    section_title("Top Random Search Trials")
    trial_candidates = [m for m, rows in search_trials_map.items() if rows]
    if trial_candidates:
        trial_model = st.selectbox(
            "Inspect CV trials for model",
            options=trial_candidates,
            key="trial_model_selector",
        )
        trial_rows = search_trials_map.get(trial_model, [])
        trial_df = pd.DataFrame(
            [
                {
                    "Rank": row["rank"],
                    "CV RMSE": row["cv_rmse"],
                    "Params": ", ".join(f"{k}={v}" for k, v in row["params"].items()),
                }
                for row in trial_rows
            ]
        )
        st.dataframe(
            trial_df.style.format({"CV RMSE": "${:,.0f}"}),
            width="stretch",
            hide_index=True,
        )
        if not trial_df.empty:
            fig_trial, ax_trial = plt.subplots(figsize=(7, 3.6))
            ax_trial.plot(trial_df["Rank"], trial_df["CV RMSE"], marker="o", linewidth=1.8)
            ax_trial.invert_xaxis()
            ax_trial.set_xlabel("Rank (1 is best)")
            ax_trial.set_ylabel("CV RMSE")
            ax_trial.set_title(f"{trial_model} random-search top trials")
            st.pyplot(fig_trial, width="stretch")

    # Custom ensemble composition
    section_title("Custom Ensemble Composition")
    base_model_names = [
        r.name for r in training["results"] if r.name != "Ensemble (Weighted Top-3)"
    ]
    default_ensemble = training["ensemble_members"]
    chosen_ensemble_models = st.multiselect(
        "Choose models for custom ensemble",
        options=base_model_names,
        default=default_ensemble,
        key="custom_ensemble_models",
    )
    if not chosen_ensemble_models:
        st.warning("Select at least one model for custom ensemble.")
        chosen_ensemble_models = default_ensemble

    ensemble_method = st.selectbox(
        "Ensemble method",
        options=[
            "Weighted Average (inverse RMSE)",
            "Simple Average",
            "Median Ensemble",
        ],
        index=0,
        key="ensemble_method_selector",
    )

    rmse_map = {
        r.name: r.rmse
        for r in training["results"]
        if r.name in chosen_ensemble_models
    }
    rmse_vals = np.array([rmse_map[m] for m in chosen_ensemble_models], dtype=float)
    custom_weight_map = {}
    if ensemble_method == "Weighted Average (inverse RMSE)":
        custom_weights = 1.0 / np.clip(rmse_vals, 1e-8, None)
        custom_weights /= custom_weights.sum()
        custom_weight_map = {
            m: float(custom_weights[i]) for i, m in enumerate(chosen_ensemble_models)
        }
    else:
        equal_w = 1.0 / len(chosen_ensemble_models)
        custom_weight_map = {m: equal_w for m in chosen_ensemble_models}

    st.markdown(
        "<p style='font-size:14px; color:#5a6480;'>"
        f"Custom ensemble uses <b>{ensemble_method}</b> across selected models.</p>",
        unsafe_allow_html=True,
    )
    ensemble_df = pd.DataFrame({
        "Model": chosen_ensemble_models,
        "Weight": [custom_weight_map[m] for m in chosen_ensemble_models],
        "RMSE": [
            leaderboard[leaderboard["Model"] == m]["RMSE"].values[0]
            for m in chosen_ensemble_models
        ],
    })
    st.dataframe(
        ensemble_df.style.format({"Weight": "{:.1%}", "RMSE": "${:,.0f}"}),
        width="stretch",
        hide_index=True,
    )

    # Recompute a live custom-ensemble metric row and merge into displayed leaderboard.
    X_test = training["X_test"]
    y_test = training["y_test"]
    member_pred_matrix = np.column_stack(
        [training["model_map"][m].predict(X_test) for m in chosen_ensemble_models]
    )
    if ensemble_method == "Weighted Average (inverse RMSE)":
        custom_preds = np.zeros(len(X_test), dtype=float)
        for i, model_name in enumerate(chosen_ensemble_models):
            custom_preds += custom_weight_map[model_name] * member_pred_matrix[:, i]
    elif ensemble_method == "Simple Average":
        custom_preds = np.mean(member_pred_matrix, axis=1)
    else:
        custom_preds = np.median(member_pred_matrix, axis=1)

    custom_rmse = float(np.sqrt(mean_squared_error(y_test, custom_preds)))
    custom_mae = float(mean_absolute_error(y_test, custom_preds))
    custom_r2 = float(r2_score(y_test, custom_preds))

    leaderboard_live = leaderboard[
        (leaderboard["Model"] != "Custom Ensemble (Current Selection)")
        & (leaderboard["Model"] != "Ensemble (Weighted Top-3)")
    ].copy()
    custom_row = pd.DataFrame(
        [
            {
                "Model": "Custom Ensemble (Current Selection)",
                "RMSE": custom_rmse,
                "MAE": custom_mae,
                "R²": custom_r2,
                "CV RMSE (5-fold)": np.nan,
                "CV RMSE Std": np.nan,
                "CV R² (5-fold)": np.nan,
            }
        ]
    )
    leaderboard_live = (
        pd.concat([leaderboard_live, custom_row], ignore_index=True)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    section_title("Model Leaderboard")
    st.dataframe(
        leaderboard_live.style
        .apply(highlight_best, axis=1)
        .format(
            {
                "RMSE": "${:,.0f}",
                "MAE": "${:,.0f}",
                "R²": "{:.4f}",
                "CV RMSE (5-fold)": "${:,.0f}",
                "CV RMSE Std": "${:,.0f}",
                "CV R² (5-fold)": "{:.4f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    section_title("Best Hyperparameters (Random Search)")
    params_map = training.get("best_params_map", {})
    params_rows = []
    for model in leaderboard_live["Model"].tolist():
        if model in params_map:
            params_rows.append(
                {
                    "Model": model,
                    "Best Params": ", ".join(
                        f"{k.replace('model__', '')}={v}" for k, v in params_map[model].items()
                    ) if params_map[model] else "No tuned params",
                }
            )
    if params_rows:
        st.dataframe(pd.DataFrame(params_rows), width="stretch", hide_index=True)

    # ── Actual vs Predicted scatter ───────────────────────────────────────────
    section_title("Actual vs Predicted — Best Model")
    best_pipe = training["best_pipeline"]
    X_test = training["X_test"]
    y_test = training["y_test"]
    preds = best_pipe.predict(X_test)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, preds, alpha=0.45, s=18, color="#2563eb", zorder=2)
    lo = min(y_test.min(), preds.min())
    hi = max(y_test.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#dc2626", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Charge ($)")
    ax.set_ylabel("Predicted Charge ($)")
    ax.set_title(f"Actual vs Predicted — {training['best_model_name']}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.legend(fontsize=10)
    st.pyplot(fig, width="stretch")
    insight_box(
        "Points on the red diagonal = perfect predictions. The cluster near the diagonal "
        "for mid-range charges is tight. The high-charge outliers (>$50k) show more variance — "
        "these are complex cases likely driven by rare conditions not captured in our features."
    )

    # ── Live prediction form ──────────────────────────────────────────────────
    section_title("Live Prediction Engine")
    st.markdown(
        "<p style='font-size:14px; color:#a5b8de; margin-bottom:20px;'>"
        "Adjust the profile below. Results update when you click Predict.</p>",
        unsafe_allow_html=True,
    )

    model_options = [r.name for r in training["results"] if r.name != "Ensemble (Weighted Top-3)"]
    model_options.append("Custom Ensemble")
    chosen = st.selectbox(
        "Select model",
        model_options,
        index=max(0, model_options.index(training["best_model_name"])) if training["best_model_name"] in model_options else 0,
    )

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        x_cols = [c for c in df.columns if c != TARGET_COL]
        half = len(x_cols) // 2
        inputs = {}

        with col1:
            for col in x_cols[:half]:
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    opts = sorted(df[col].dropna().unique().tolist())
                    inputs[col] = st.selectbox(col.upper(), opts)
                elif np.issubdtype(df[col].dtype, np.integer):
                    lo, hi = int(df[col].min()), int(df[col].max())
                    inputs[col] = st.slider(col.upper(), lo, hi, int(df[col].median()))
                else:
                    lo, hi = float(df[col].min()), float(df[col].max())
                    inputs[col] = st.slider(col.upper(), lo, hi, float(df[col].median()), 0.1)

        with col2:
            for col in x_cols[half:]:
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    opts = sorted(df[col].dropna().unique().tolist())
                    inputs[col] = st.selectbox(col.upper(), opts)
                elif np.issubdtype(df[col].dtype, np.integer):
                    lo, hi = int(df[col].min()), int(df[col].max())
                    inputs[col] = st.slider(col.upper(), lo, hi, int(df[col].median()))
                else:
                    lo, hi = float(df[col].min()), float(df[col].max())
                    inputs[col] = st.slider(col.upper(), lo, hi, float(df[col].median()), 0.1)

        predict_btn = st.form_submit_button("⚡ Predict Insurance Charges", width="stretch")

    if predict_btn:
        input_df = pd.DataFrame([inputs])
        model_map = training["model_map"]

        if chosen == "Custom Ensemble":
            member_preds = np.array(
                [model_map[m].predict(input_df)[0] for m in chosen_ensemble_models],
                dtype=float,
            )
            if ensemble_method == "Weighted Average (inverse RMSE)":
                prediction = sum(custom_weight_map[m] * model_map[m].predict(input_df)[0] for m in chosen_ensemble_models)
            elif ensemble_method == "Simple Average":
                prediction = float(np.mean(member_preds))
            else:
                prediction = float(np.median(member_preds))
        else:
            prediction = model_map[chosen].predict(input_df)[0]

        avg = df[TARGET_COL].mean()
        pct = ((prediction - avg) / avg) * 100
        pct_str = f"{pct:+.1f}% vs average"

        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%);
                border-radius: 16px;
                padding: 28px 32px;
                margin-top: 12px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            ">
                <div>
                    <div style="font-size: 12px; font-weight: 600; color: #93c5fd;
                                text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;">
                        Predicted Annual Insurance Charge
                    </div>
                    <div style="font-size: 42px; font-weight: 700; color: #ffffff;
                                letter-spacing: -1px; font-family: 'DM Mono', monospace;">
                        ${prediction:,.2f}
                    </div>
                    <div style="font-size: 14px; color: #93c5fd; margin-top: 8px;">
                        {pct_str} · Model: {chosen}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; color: #93c5fd; margin-bottom: 4px;">
                        Portfolio average
                    </div>
                    <div style="font-size: 24px; font-weight: 600; color: #bfdbfe;">
                        ${avg:,.0f}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
