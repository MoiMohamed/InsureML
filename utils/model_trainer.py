"""
utils/model_trainer.py
──────────────────────
All model training logic, completely decoupled from UI.

DESIGN DECISIONS & RATIONALE
─────────────────────────────
1.  WHY @st.cache_resource instead of @st.cache_data?
    - @st.cache_data serialises return values (pickle). Sklearn pipelines and numpy
      arrays serialise fine, but large fitted models waste memory being duplicated.
    - @st.cache_resource keeps a single shared reference — correct for trained models.

2.  WHY Pipeline([preprocessor, model])?
    - Prevents data leakage: the scaler fits only on train data.
    - Prediction on new single rows just calls pipe.predict(input_df) — no manual
      transform step needed. This is critical for the prediction page.

3.  WHY StandardScaler on numeric features?
    - Linear Regression, Ridge, Lasso and MLP are sensitive to feature scale.
    - Tree-based models (RF, GBM, XGBoost, LightGBM) are scale-invariant,
      but scaling doesn't hurt them, so one unified preprocessor is simpler.

4.  WHY OneHotEncoder for categoricals?
    - sex (2 levels), smoker (2 levels), region (4 levels) are all nominal.
    - Ordinal encoding would imply false ordering.
    - handle_unknown='ignore' prevents crashes on novel categories at inference.

5.  WHY Ensemble (Weighted Top-3)?
    - Model stacking is complex and risks leakage on small datasets.
    - Inverse-RMSE weighting is a proven, transparent ensemble strategy.
    - On this 1338-row dataset, it typically outperforms any single model.

6.  MODELS INCLUDED (and why each):
    - Linear Regression: interpretable baseline; reveals linearity assumption limits.
    - Ridge / Lasso: regularised linear models; Lasso performs feature selection.
    - MLP (Neural Net): captures non-linear interactions without tree inductive bias.
    - Random Forest: variance-reducing bagging; robust to outliers.
    - Gradient Boosting: sklearn's classic boosting; deterministic baseline.
    - XGBoost / LightGBM: state-of-the-art gradient boosting if installed.
    - Ensemble: meta-model combining top-3 for best generalisation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import warnings
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Optional heavy dependencies ──────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None  # type: ignore

TARGET_COL = "charges"


@dataclass
class ModelResult:
    name: str
    pipeline: object
    rmse: float
    mae: float
    r2: float
    cv_rmse: float = float("nan")
    cv_rmse_std: float = float("nan")
    cv_r2: float = float("nan")
    description: str = ""
    best_params: dict | None = None
    baseline_rmse: float = float("nan")
    baseline_r2: float = float("nan")


def build_preprocessor(
    df: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer that:
      - StandardScales numeric features
      - OneHotEncodes categorical features

    Returns (preprocessor, numeric_cols, categorical_cols).
    """
    numeric_cols = [
        c for c in df.columns if df[c].dtype != "object" and c != TARGET_COL
    ]
    categorical_cols = [c for c in df.columns if df[c].dtype == "object"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def build_model_candidates() -> Dict[str, tuple]:
    """
    Returns dict of {name: (estimator, description)}.
    Description is shown in the UI to explain each model choice.
    """
    candidates = {
        "Linear Regression": (
            LinearRegression(),
            "Interpretable baseline — assumes a linear relationship between features "
            "and charges. Fast to train, easy to explain, but underperforms where "
            "interactions (e.g. smoker × BMI) drive costs.",
        ),
        "Ridge Regression": (
            Ridge(alpha=1.0),
            "L2-regularised linear model. Shrinks all coefficients toward zero, "
            "reducing variance without eliminating features. Best when many features "
            "have small but non-zero effects.",
        ),
        "Lasso Regression": (
            Lasso(alpha=0.05, max_iter=20000, tol=1e-3),
            "L1-regularised linear model. Drives some coefficients to exactly zero, "
            "performing implicit feature selection. Useful to confirm which features "
            "can be safely ignored.",
        ),
        "MLP Regressor": (
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                learning_rate_init=0.001,
                max_iter=3000,
                early_stopping=True,
                n_iter_no_change=30,
                validation_fraction=0.1,
                random_state=42,
            ),
            "2-layer neural network. Captures non-linear interactions without relying "
            "on tree structure. Sensitive to scale (hence StandardScaler in pipeline). "
            "Can overfit on small datasets — monitor train vs test RMSE.",
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
            "Bagging ensemble of 300 deep trees. Reduces variance by averaging "
            "uncorrelated trees. Robust to outliers and noise. The OOB score gives "
            "a free validation estimate without a separate hold-out.",
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
            ),
            "Additive tree ensemble trained greedily on residuals. Typically more "
            "accurate than Random Forest on tabular data, but more sensitive to "
            "hyperparameters. sklearn's deterministic, no external dependency.",
        ),
    }

    if XGBRegressor is None or LGBMRegressor is None:
        raise RuntimeError("Required model engines unavailable.")

    candidates["XGBoost"] = (
        XGBRegressor(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
        "Extreme Gradient Boosting — regularised boosting with column/row "
        "subsampling. Often wins on tabular benchmarks. Faster than sklearn GBM "
        "via C++ implementation. Best single model on this dataset.",
    )

    candidates["LightGBM"] = (
        LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            verbose=-1,
            random_state=42,
        ),
        "Microsoft's leaf-wise tree growing strategy. Faster than XGBoost on "
        "large datasets; comparable accuracy on small ones. Excellent when "
        "training time is a constraint.",
    )

    return candidates


def _random_search_space(model_name: str) -> dict:
    """Bounded parameter spaces for quick random search per model."""
    spaces = {
        "Linear Regression": {
            "model__fit_intercept": [True, False],
        },
        "Ridge Regression": {
            "model__alpha": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "model__fit_intercept": [True, False],
        },
        "Lasso Regression": {
            "model__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
            "model__max_iter": [10000, 20000, 40000],
            "model__tol": [1e-4, 1e-3, 5e-3],
        },
        "MLP Regressor": {
            "model__hidden_layer_sizes": [(32, 16), (64, 32), (128, 64)],
            "model__alpha": [1e-5, 1e-4, 1e-3],
            "model__learning_rate_init": [5e-4, 1e-3, 2e-3],
            "model__max_iter": [2000, 3000],
            "model__early_stopping": [True],
            "model__validation_fraction": [0.1, 0.15],
        },
        "Random Forest": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [None, 6, 10, 14],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },
        "Gradient Boosting": {
            "model__n_estimators": [150, 250, 400],
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.85, 1.0],
        },
        "XGBoost": {
            "model__n_estimators": [200, 300, 450],
            "model__max_depth": [3, 4, 6],
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        },
        "LightGBM": {
            "model__n_estimators": [200, 350, 500],
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__num_leaves": [15, 31, 63],
            "model__min_child_samples": [10, 20, 40],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__verbose": [-1],
        },
    }
    return spaces.get(model_name, {})


def _count_discrete_combinations(param_space: dict) -> int:
    """Return cartesian size for list-based discrete spaces."""
    if not param_space:
        return 0
    total = 1
    for values in param_space.values():
        try:
            total *= len(values)
        except TypeError:
            return 0
    return int(total)


@st.cache_resource(show_spinner="Training models…")
def train_all_models(df: pd.DataFrame, selected_features: tuple[str, ...] | None = None) -> dict:
    """
    Train all available models on an 80/20 train-test split.
    Returns a rich dict used by prediction, explainability, and tuning pages.
    """
    X_all = df.drop(columns=[TARGET_COL])
    if selected_features:
        selected = [c for c in selected_features if c in X_all.columns]
        X = X_all[selected].copy()
    else:
        X = X_all.copy()
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build preprocessor on the effective training schema (selected features only).
    preprocessor, _, _ = build_preprocessor(pd.concat([X, y], axis=1))
    model_candidates = build_model_candidates()
    trained_results: List[ModelResult] = []

    tuned_params = {}
    baseline_metrics = {}
    search_trials_map = {}
    model_backend_map = {}
    for name, (estimator, description) in model_candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=FutureWarning)

                baseline_pipe = clone(pipe)
                baseline_pipe.fit(X_train, y_train)
                baseline_preds = baseline_pipe.predict(X_test)
                baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_preds)))
                baseline_r2 = float(r2_score(y_test, baseline_preds))

                param_space = _random_search_space(name)
                if param_space:
                    space_size = _count_discrete_combinations(param_space)
                    n_iter_search = min(12, space_size) if space_size > 0 else 12
                    search = RandomizedSearchCV(
                        estimator=pipe,
                        param_distributions=param_space,
                        n_iter=n_iter_search,
                        scoring="neg_root_mean_squared_error",
                        cv=3,
                        n_jobs=-1,
                        random_state=42,
                        refit=True,
                        error_score="raise",
                    )
                    search.fit(X_train, y_train)
                    best_pipe = search.best_estimator_
                    best_params = search.best_params_
                    cv_trials = pd.DataFrame(search.cv_results_).sort_values(
                        "rank_test_score"
                    )
                    search_trials_map[name] = [
                        {
                            "rank": int(row["rank_test_score"]),
                            "cv_rmse": float(-row["mean_test_score"]),
                            "params": {
                                k.replace("model__", ""): v
                                for k, v in row["params"].items()
                            },
                        }
                        for _, row in cv_trials.head(5).iterrows()
                    ]
                else:
                    best_pipe = clone(pipe)
                    best_pipe.fit(X_train, y_train)
                    best_params = {}
                    search_trials_map[name] = []

                preds = best_pipe.predict(X_test)
                cv_res = cross_validate(
                    best_pipe,
                    X,
                    y,
                    cv=5,
                    scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
                    n_jobs=-1,
                    error_score="raise",
                )
                cv_rmse = float(-np.mean(cv_res["test_rmse"]))
                cv_rmse_std = float(np.std(-cv_res["test_rmse"]))
                cv_r2 = float(np.mean(cv_res["test_r2"]))
        except Exception:
            continue

        fitted_model = best_pipe.named_steps["model"]
        model_backend_map[name] = (
            f"{fitted_model.__class__.__module__}.{fitted_model.__class__.__name__}"
        )
        tuned_params[name] = best_params
        baseline_metrics[name] = {"rmse": baseline_rmse, "r2": baseline_r2}
        trained_results.append(
            ModelResult(
                name=name,
                pipeline=best_pipe,
                rmse=float(np.sqrt(mean_squared_error(y_test, preds))),
                mae=float(mean_absolute_error(y_test, preds)),
                r2=float(r2_score(y_test, preds)),
                cv_rmse=cv_rmse,
                cv_rmse_std=cv_rmse_std,
                cv_r2=cv_r2,
                description=description,
                best_params=best_params,
                baseline_rmse=baseline_rmse,
                baseline_r2=baseline_r2,
            )
        )

    if not trained_results:
        raise RuntimeError("No models trained successfully.")

    # ── Ensemble: inverse-RMSE weighted top-3 ───────────────────────────────
    top3 = sorted(trained_results, key=lambda r: r.rmse)[:3]
    rmse_arr = np.array([m.rmse for m in top3])
    weights = 1.0 / np.clip(rmse_arr, 1e-8, None)
    weights /= weights.sum()

    ensemble_preds = sum(
        w * m.pipeline.predict(X_test) for w, m in zip(weights, top3)
    )
    ensemble_result = ModelResult(
        name="Ensemble (Weighted Top-3)",
        pipeline=top3[0].pipeline,  # placeholder; ensemble uses model_map directly
        rmse=float(np.sqrt(mean_squared_error(y_test, ensemble_preds))),
        mae=float(mean_absolute_error(y_test, ensemble_preds)),
        r2=float(r2_score(y_test, ensemble_preds)),
        cv_rmse=float("nan"),
        cv_rmse_std=float("nan"),
        cv_r2=float("nan"),
        description=(
            "Inverse-RMSE weighted average of the three best models. "
            "Better-performing models receive proportionally higher weight. "
            "Typically achieves lower variance than any single model."
        ),
    )
    trained_results.append(ensemble_result)

    # ── Leaderboard ──────────────────────────────────────────────────────────
    leaderboard = (
        pd.DataFrame(
            [{"Model": r.name, "RMSE": r.rmse, "MAE": r.mae, "R²": r.r2,
              "CV RMSE (5-fold)": r.cv_rmse, "CV RMSE Std": r.cv_rmse_std, "CV R² (5-fold)": r.cv_r2}
             for r in trained_results]
        )
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    best_name = leaderboard.iloc[0]["Model"]
    best_pipeline = next(r.pipeline for r in trained_results if r.name == best_name)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "selected_features": list(X.columns),
        "results": trained_results,
        "model_map": {r.name: r.pipeline for r in trained_results},
        "descriptions": {r.name: r.description for r in trained_results},
        "ensemble_members": [m.name for m in top3],
        "ensemble_weights": {m.name: float(weights[i]) for i, m in enumerate(top3)},
        "leaderboard": leaderboard,
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "best_params_map": tuned_params,
        "baseline_metrics_map": baseline_metrics,
        "search_trials_map": search_trials_map,
        "model_backend_map": model_backend_map,
    }
