"""
pages/p05_tuning.py
────────────────────
Page 5: Hyperparameter Tuning + Weights & Biases Integration

TECHNICAL DECISIONS:
  - WHY GridSearchCV?
    Exhaustive grid search with cross-validation guarantees we evaluate every
    parameter combination on multiple folds, not just a lucky split. For small
    grids (<100 combinations) it's faster than Bayesian optimisation to code
    and just as effective.

  - WHY cv=5?
    5-fold CV is the community standard. 3-fold has high variance on 1338 rows.
    10-fold is slower without meaningfully reducing variance further.

  - WHY negative RMSE scoring?
    sklearn convention: all scorers must be "higher = better" so RMSE becomes
    neg_root_mean_squared_error. We negate the output to recover actual RMSE.

  - WHY W&B?
    Local GridSearchCV results vanish when you close the app. W&B persists
    every run, enables comparison across sessions, and produces publication-
    quality parallel coordinates plots for hyperparameter analysis.
    It also lets multiple team members run experiments asynchronously.

  - WHY return_train_score=True?
    Comparing train vs test score diagnoses overfitting. If train RMSE is
    much lower than test RMSE, the model is overfitting — increase regularisation.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from utils.model_trainer import build_model_candidates, build_preprocessor
from utils.ui_components import insight_box, page_header, section_title, tech_decision_box

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

TARGET_COL = "charges"

PARAM_GRIDS = {
    "Linear Regression": {},
    "Ridge Regression": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "Lasso Regression": {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
    "MLP Regressor": {
        "model__hidden_layer_sizes": [(32, 16), (64, 32), (128, 64)],
        "model__alpha": [0.0001, 0.001, 0.01],
        "model__learning_rate_init": [0.001, 0.005],
    },
    "Random Forest": {
        "model__n_estimators": [100, 300],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [3, 4, 5],
    },
    "XGBoost": {
        "model__n_estimators": [200, 350],
        "model__max_depth": [3, 4, 6],
        "model__learning_rate": [0.03, 0.05],
        "model__subsample": [0.8, 1.0],
    },
    "LightGBM": {
        "model__n_estimators": [200, 400],
        "model__num_leaves": [31, 63],
        "model__learning_rate": [0.03, 0.05],
    },
}

PARAM_DESCRIPTIONS = {
    "model__alpha": "Regularisation strength. Higher = more penalty on large coefficients.",
    "model__hidden_layer_sizes": "Neural network architecture. (64,32) = 2 hidden layers.",
    "model__learning_rate_init": "Initial learning rate for gradient descent.",
    "model__n_estimators": "Number of trees. More trees = lower variance but slower.",
    "model__max_depth": "Max depth per tree. Deeper = captures more interactions, risk of overfitting.",
    "model__learning_rate": "Shrinkage factor. Lower rate needs more estimators but generalises better.",
    "model__subsample": "Fraction of training data per tree. <1.0 = stochastic boosting.",
    "model__num_leaves": "LightGBM leaf count. Controls model complexity.",
    "model__min_samples_split": "Minimum samples to split a node. Higher = more regularisation.",
}


def _wandb_status() -> tuple[bool, str]:
    if not WANDB_AVAILABLE:
        return False, "wandb not installed — run `pip install wandb`"
    if not os.environ.get("WANDB_API_KEY"):
        return False, "WANDB_API_KEY not set — run `export WANDB_API_KEY=your_key`"
    return True, "W&B connected ✓"


def render(df: pd.DataFrame, training: dict):
    page_header(
        title="Hyperparameter Tuning",
        subtitle="GridSearchCV + 5-fold CV · Logged to Weights & Biases",
        emoji="🧪",
    )

    # ── W&B status ───────────────────────────────────────────────────────────
    wandb_ready, wandb_msg = _wandb_status()
    if wandb_ready:
        st.success(f"🟢 {wandb_msg}")
    else:
        st.warning(f"🟡 {wandb_msg}")
        st.markdown(
            """
            <div style="background:#fffbeb; border:1px solid #fcd34d; border-radius:10px;
                        padding:14px 18px; font-size:13px; color:#92400e; margin-bottom:16px;">
                <strong>To enable W&B logging:</strong><br>
                1. Create account at <code>wandb.ai</code><br>
                2. Run <code>wandb login</code> in terminal<br>
                3. Or set <code>export WANDB_API_KEY=your_key_here</code><br>
                Tuning will still run locally without W&B.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Technical decisions ───────────────────────────────────────────────────
    section_title("Architecture Decisions")
    tech_decision_box(
        "GridSearchCV with cv=5",
        "Evaluates every hyperparameter combination on 5 train/validation folds. "
        "Score is averaged across folds to reduce split variance. "
        "Preferred over random search for small grids (<200 combinations).",
    )

    # ── Parameter grid explanations ───────────────────────────────────────────
    section_title("Hyperparameter Glossary")
    for param, explanation in PARAM_DESCRIPTIONS.items():
        st.markdown(
            f"""
            <div style="display:flex; gap:12px; padding:10px 0;
                        border-bottom:1px solid #e8eaf0; font-size:13.5px;">
                <div style="min-width:220px; font-family:'DM Mono',monospace;
                             color:#2563eb; font-size:12px; padding-top:1px;">
                    {param}
                </div>
                <div style="color:#5a6480;">{explanation}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Model selector ────────────────────────────────────────────────────────
    section_title("Run Tuning Experiment")
    model_candidates = build_model_candidates()
    available_models = [k for k in PARAM_GRIDS.keys() if k in model_candidates]

    c1, c2 = st.columns([2, 1])
    with c1:
        model_choice = st.selectbox("Select model to tune", available_models)
    with c2:
        project_name = st.text_input("W&B project name", value="insurance-ml")

    # Show the grid that will be searched
    grid = PARAM_GRIDS[model_choice]
    if grid:
        n_combinations = 1
        for v in grid.values():
            n_combinations *= len(v)
        st.markdown(
            f"""
            <div style="background:#f0f4ff; border:1px solid #c7d4f8; border-radius:10px;
                        padding:14px 18px; font-size:13px; color:#1a3060; margin-bottom:16px;">
                <strong>Grid to search:</strong> {n_combinations} combinations
                (5-fold CV = {n_combinations * 5} model fits)<br>
                <span style="font-family:'DM Mono',monospace; font-size:12px; color:#2563eb;">
                    {str(grid)}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(f"{model_choice} has no hyperparameters to tune — will run a single CV evaluation.")

    run_tuning = st.button("🚀 Run Tuning Experiment", width="stretch", type="primary")

    if not run_tuning:
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    preprocessor, _, _ = build_preprocessor(df)
    estimator, _ = model_candidates[model_choice]
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    if WANDB_AVAILABLE and wandb_ready:
        wandb.init(project=project_name, job_type="hyperparameter-tuning",
                   config={"model": model_choice, "param_grid": grid, "cv": 5})

    progress = st.progress(0, text="Fitting folds…")

    with st.spinner(f"Running GridSearchCV for {model_choice}…"):
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid if grid else [{}],
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )
        search.fit(X, y)

    progress.progress(100, text="Done!")

    # ── Results display ───────────────────────────────────────────────────────
    section_title("Tuning Results")
    results_df = pd.DataFrame(search.cv_results_)
    results_df["cv_rmse"] = -results_df["mean_test_score"]
    results_df["train_rmse"] = -results_df["mean_train_score"]
    results_df["overfit_gap"] = results_df["train_rmse"] - results_df["cv_rmse"]

    show_cols = ["params", "cv_rmse", "train_rmse", "overfit_gap",
                 "std_test_score", "rank_test_score"]
    display = results_df[[c for c in show_cols if c in results_df.columns]].copy()
    display = display.sort_values("rank_test_score")

    st.dataframe(
        display.head(15).style.format({
            "cv_rmse": "${:,.0f}",
            "train_rmse": "${:,.0f}",
            "overfit_gap": "${:,.0f}",
            "std_test_score": "{:.1f}",
        }),
        width="stretch",
        hide_index=True,
    )

    best_rmse = -search.best_score_
    best_params = search.best_params_

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #f0fdf4, #dcfce7);
            border: 1px solid #86efac;
            border-radius: 12px;
            padding: 20px 24px;
            margin-top: 12px;
        ">
            <div style="font-size: 13px; font-weight: 600; color: #166534;
                        text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px;">
                Best Configuration Found
            </div>
            <div style="font-size: 28px; font-weight: 700; color: #14532d;
                        font-family: 'DM Mono', monospace; margin-bottom: 8px;">
                RMSE: ${best_rmse:,.2f}
            </div>
            <div style="font-size: 13px; color: #166534; font-family: 'DM Mono', monospace;">
                {str(best_params)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── RMSE chart ────────────────────────────────────────────────────────────
    if len(display) > 1:
        import matplotlib.pyplot as plt

        section_title("CV RMSE Across Parameter Combinations")
        fig, ax = plt.subplots(figsize=(9, 3.5))
        x = range(len(display.head(15)))
        ax.bar(x, display["cv_rmse"].head(15), color="#2563eb", alpha=0.75,
               width=0.6, label="CV RMSE")
        if "train_rmse" in display.columns:
            ax.plot(x, display["train_rmse"].head(15), "o--",
                    color="#dc2626", linewidth=1.5, markersize=5, label="Train RMSE")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"#{i+1}" for i in x], fontsize=9)
        ax.set_xlabel("Parameter combination (ranked by CV RMSE)")
        ax.set_ylabel("RMSE ($)")
        ax.set_title("CV vs Train RMSE — Top 15 Combinations")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))
        ax.legend()
        st.pyplot(fig, width="stretch")

        insight_box(
            "If Train RMSE is much lower than CV RMSE (large gap), the model is overfitting. "
            "Increase regularisation, reduce tree depth, or reduce n_estimators. "
            "If both are similarly high, the model is underfitting — increase complexity."
        )

    # ── W&B logging ───────────────────────────────────────────────────────────
    if WANDB_AVAILABLE and wandb_ready:
        wandb.log({
            "best_cv_rmse": best_rmse,
            "best_params": str(best_params),
            "model": model_choice,
        })
        table = wandb.Table(dataframe=display.head(30).astype(str))
        wandb.log({"tuning_results": table})
        wandb.finish()
        st.success("✅ Experiment logged to Weights & Biases.")
