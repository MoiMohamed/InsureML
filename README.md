# 🧬 InsureML — Insurance Cost Intelligence App

A production-grade Streamlit app predicting medical insurance charges using 7+ ML models.

## Quick Start

```bash
# 1. Clone / copy project
cd insurance_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Place dataset
cp your_insurance.csv data/insurance.csv
# Or it auto-downloads from GitHub on first run

# 4. (Optional) Set W&B API key
export WANDB_API_KEY=your_key_here

# 5. Run
streamlit run app.py
```

## Project Structure

```
insurance_app/
│
├── app.py                    # Entry point — page routing + sidebar
│
├── utils/
│   ├── data_loader.py        # Dataset loading with 4 fallback strategies
│   ├── model_trainer.py      # All model training + ensemble logic
│   └── ui_components.py      # Reusable styled components
│
├── pages/
│   ├── p01_business.py       # Business case + data dictionary
│   ├── p02_data_viz.py       # EDA + pricing simulator
│   ├── p03_prediction.py     # Model leaderboard + live prediction
│   ├── p04_explainability.py # SHAP + permutation importance
│   └── p05_tuning.py         # GridSearchCV + W&B logging
│
├── data/
│   └── insurance.csv         # (not committed — auto-downloaded)
│
├── artifacts/                # Exported model files (.joblib)
└── requirements.txt
```

## Models Included

| Model | Why |
|---|---|
| Linear Regression | Interpretable baseline |
| Ridge Regression | L2 regularisation |
| Lasso Regression | L1 + feature selection |
| MLP Neural Network | Non-linear interactions |
| Random Forest | Bagging, robust to noise |
| Gradient Boosting | Boosting, sklearn native |
| XGBoost *(optional)* | State-of-the-art boosting |
| LightGBM *(optional)* | Fast leaf-wise boosting |
| Ensemble (Top-3) | Inverse-RMSE weighted avg |

## Key Technical Decisions

- **Pipeline**: preprocessor + model in one object → no data leakage
- **StandardScaler**: required for linear models + MLP
- **OneHotEncoder**: nominal categoricals (sex, smoker, region)
- **80/20 split**, `random_state=42` for reproducibility
- **SHAP**: auto-selects TreeExplainer / LinearExplainer / KernelExplainer
- **Ensemble**: inversely-weighted by RMSE, top-3 models

## Deployment

Works on Streamlit Cloud and HuggingFace Spaces. Add secrets for `WANDB_API_KEY`.
