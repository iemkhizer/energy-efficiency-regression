import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
import joblib
import keras
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():
    model   = keras.models.load_model(APP_DIR / "model.keras")
    scaler  = joblib.load(APP_DIR / "scaler.joblib")
    columns = joblib.load(APP_DIR / "feature_columns.joblib")
    return model, scaler, columns

model, scaler, columns = load_artifacts()

st.title("Building Energy Load Predictor")
st.success("Model loaded")

DATA_FILE = APP_DIR.parent / "data" / "energy_efficiency_data.csv"

@st.cache_data
def load_shapes():
    df = pd.read_csv(DATA_FILE)
    return df.groupby("Relative_Compactness")[["Wall_Area", "Roof_Area", "Overall_Height"]].first()

shapes = load_shapes()

ORIENT = {"North": 2, "East": 3, "South": 4, "West": 5}
DIST   = {"Uniform": 1, "North": 2, "East": 3, "South": 4, "West": 5}

rc  = st.selectbox("Building shape (relative compactness)", shapes.index.tolist())
geo = shapes.loc[rc]
st.caption(f"Height {geo['Overall_Height']} m · wall {geo['Wall_Area']} m² · roof {geo['Roof_Area']} m²")

orientation  = st.selectbox("Orientation", list(ORIENT.keys()))
glazing      = st.slider("Glazing area (share of floor area)", 0.0, 0.40, 0.25, 0.05)
distribution = st.selectbox("Glazing distribution", list(DIST.keys()), disabled=(glazing == 0))

def predict_building(building):
    new = pd.DataFrame([building])
    new_X = pd.get_dummies(new, columns=["Orientation", "Glazing_Area_Distribution"], dtype=int)
    new_X = new_X.reindex(columns=columns, fill_value=0)
    new_scaled = scaler.transform(new_X)
    pred = model.predict(new_scaled, verbose=0)
    return float(pred[0, 0]), float(pred[0, 1])

if st.button("Predict"):
    building = {
        "Relative_Compactness": rc,
        "Wall_Area": geo["Wall_Area"],
        "Roof_Area": geo["Roof_Area"],
        "Overall_Height": geo["Overall_Height"],
        "Orientation": ORIENT[orientation],
        "Glazing_Area": glazing,
        "Glazing_Area_Distribution": 0 if glazing == 0 else DIST[distribution],
    }
    heat, cool = predict_building(building)

    c1, c2 = st.columns(2)
    c1.metric("Heating load", f"{heat:.2f} kWh/m²")
    c2.metric("Cooling load", f"{cool:.2f} kWh/m²")