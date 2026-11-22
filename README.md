# Energy Efficiency Regression

Assignment 2 for **Intro to Data Science (IT803A)**, University of Skövde.

This project predicts a building's **Heating Load** and **Cooling Load** (kWh/m²) from 8 design features. It compares classical regression models with a neural network, tracks experiments with MLflow, tunes the network with GridSearchCV, and deploys the final model as a Streamlit app.

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

## Results (test set, 154 buildings)

| Model | R² Heating | R² Cooling | MSE Heating | MSE Cooling |
|---|---|---|---|---|
| Linear Regression | 0.921 | 0.895 | 8.25 | 9.73 |
| Ridge Regression | 0.920 | 0.894 | 8.35 | 9.79 |
| Lasso Regression | 0.902 | 0.883 | 10.19 | 10.80 |
| Neural Network (64-32, lr 0.001) | 0.996 | 0.985 | 0.42 | 1.41 |
| **Neural Network, GridSearchCV best (64-32, lr 0.01)** | **0.995** | **0.989** | **0.49** | **1.03** |

The GridSearchCV model is the deployed model.

## Project structure

```
├── DS25-AS2-ID.ipynb              # Main notebook (submission)
├── data/
│   └── energy_efficiency_data.csv
├── app/
│   ├── app.py                     # Streamlit app (Part 4)
│   ├── model.keras                # Final neural network
│   ├── scaler.joblib              # StandardScaler fitted on the training set
│   └── feature_columns.joblib     # The 13 model input columns, in training order
└── requirements.txt
```

MLflow stores its tracking data locally in `mlflow.db` and `mlruns/` (not committed).

## Progress

- [x] Part 1: Data preparation & EDA
- [x] Part 2: Classical regression models (Linear, Ridge, Lasso)
- [x] Part 3: Neural network regression + MLflow + GridSearchCV
- [x] Part 4: Deployment (Streamlit)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook DS25-AS2-ID.ipynb
```

## Run the app

```bash
streamlit run app/app.py
```

Then open http://localhost:8501.

## View the MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.
