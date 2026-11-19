# Energy Efficiency Regression

Assignment 2 for **Intro to Data Science (IT803A)**, University of Skövde.

This project predicts a building's **Heating Load** and **Cooling Load** (kWh/m²) from 8 design features. It compares classical regression models with a neural network, tracks experiments with MLflow, and deploys the trained model.

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

## Project structure

```
├── DS25-AS2-ID.ipynb          # Main notebook (submission)
├── data/
│   └── energy_efficiency_data.csv
├── app/                       # Deployment app (Part 4)
└── requirements.txt
```

## Progress

- [x] Part 1: Data preparation & EDA
- [x] Part 2: Classical regression models (Linear, Ridge, Lasso)
- [ ] Part 3: Neural network regression + MLflow + GridSearchCV
- [ ] Part 4: Deployment

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook DS25-AS2-ID.ipynb
```
