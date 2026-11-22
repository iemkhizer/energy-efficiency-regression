# Building Energy Load Prediction

Predicts the **heating load** and **cooling load** (kWh/m²) of a residential building from its design: shape, height, orientation and windows. The project runs end to end, from exploratory analysis and classical baselines to a tuned neural network that is deployed as a live web app.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://energy-efficiency-regression.streamlit.app/)

**Live demo:** https://energy-efficiency-regression.streamlit.app/

## Highlights

- The neural network reaches **R² 0.995 (heating)** and **0.989 (cooling)** on unseen buildings, compared with 0.921 / 0.895 for linear regression. That cuts the test MSE by **94%** for heating and **89%** for cooling.
- The exploratory analysis explains why linear models fall short: some feature effects reverse once building height is taken into account (**Simpson's paradox**), and orientation affects cooling much more for some building shapes than for others (an **interaction**).
- `Surface_Area` was removed after a **VIF** check, because it is an exact combination of the other areas (wall + 2 × roof).
- Hyperparameters were chosen with **GridSearchCV** (3-fold cross-validation on the training set), not on the test set.
- Every experiment is tracked in **MLflow**, and the final model is served with **Streamlit**.

## Tech stack

Python · pandas · NumPy · scikit-learn · TensorFlow / Keras · SciKeras · MLflow · Streamlit · Matplotlib · Seaborn

## Dataset

[UCI Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency): 768 residential buildings simulated in Ecotect (12 base shapes × 4 orientations × glazing settings).
A. Tsanas & A. Xifara (2012), *Energy and Buildings* 49, 560–567.

| Feature | Description |
|---|---|
| Relative_Compactness | Volume-to-surface ratio relative to a cube |
| Surface_Area, Wall_Area, Roof_Area | Envelope areas (m²) |
| Overall_Height | 3.5 m or 7 m |
| Orientation | 2–5 (N, E, S, W) |
| Glazing_Area | Window area as a fraction of floor area (0, 0.1, 0.25, 0.4) |
| Glazing_Area_Distribution | Where the windows are placed (0 = none, 1–5) |
| **Heating_Load, Cooling_Load** | Targets (kWh/m²) |

## Approach

1. **Exploratory analysis:** distributions and skew, IQR outlier check (including comparisons between otherwise identical buildings), correlations, mean-load plots per feature, and multicollinearity (VIF).
2. **Preprocessing:** dropped `Surface_Area`, one-hot encoded `Orientation` and `Glazing_Area_Distribution` (13 input features), split 80/20 (614 train / 154 test), and fitted `StandardScaler` on the training set only to avoid data leakage.
3. **Baselines:** Linear, Ridge and Lasso regression, evaluated with MSE and R² for each target.
4. **Neural network:** a Keras MLP (13 → 64 → 32 → 2, ReLU, Adam, MSE loss) that predicts both loads at once, trained with early stopping (patience 200, best weights restored).
5. **Tuning:** GridSearchCV through SciKeras over hidden layers {(32, 16), (64, 32)} × learning rate {0.001, 0.01}. Best: (64, 32) with learning rate 0.01.
6. **Experiment tracking:** parameters, metrics, learning curves and models logged to MLflow, with the grid search stored as nested runs.
7. **Deployment:** the model, the fitted scaler and the training column order are saved, so the app applies exactly the same preprocessing to new inputs.

## Results (test set, 154 buildings)

| Model | R² Heating | R² Cooling | MSE Heating | MSE Cooling |
|---|---|---|---|---|
| Linear Regression | 0.921 | 0.895 | 8.25 | 9.73 |
| Ridge Regression | 0.920 | 0.894 | 8.35 | 9.79 |
| Lasso Regression | 0.902 | 0.884 | 10.19 | 10.80 |
| Neural Network (64-32, lr 0.001) | 0.996 | 0.985 | 0.42 | 1.41 |
| **Neural Network, GridSearchCV best (64-32, lr 0.01)** | **0.995** | **0.989** | **0.49** | **1.03** |

The tuned model is the deployed one. It was selected by cross-validation, and the test set was used only for the final comparison.

## The app

The user picks one of the 12 building shapes from the dataset (height, wall area and roof area fill in automatically, because they are fixed for each shape), then the orientation, glazing area and glazing distribution. The app returns the predicted heating and cooling loads. Limiting the geometry to real shapes keeps the model from being asked to extrapolate.

## Project structure

```
├── DS25-AS2-ID.ipynb              # Full analysis: EDA, models, MLflow, tuning, deployment
├── data/
│   └── energy_efficiency_data.csv
├── app/
│   ├── app.py                     # Streamlit app
│   ├── model.keras                # Final neural network
│   ├── scaler.joblib              # StandardScaler fitted on the training set
│   ├── feature_columns.joblib     # The 13 model input columns, in training order
│   └── requirements.txt           # Minimal dependencies for the hosted app
└── requirements.txt               # Full environment for the notebook
```

MLflow stores its tracking data locally in `mlflow.db` and `mlruns/` (not committed).

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook DS25-AS2-ID.ipynb
```

Start the app:

```bash
streamlit run app/app.py
```

Then open http://localhost:8501.

View the MLflow experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.

## Author

**Mohammad Khizar Imran** · [GitHub](https://github.com/iemkhizer)
