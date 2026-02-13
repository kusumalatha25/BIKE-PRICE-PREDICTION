import streamlit as st
import pandas as pd
import pickle
import base64
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Bike Price Predictor", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("RF_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- SAFE BACKGROUND IMAGE ----------------
def set_bg(image_path):
    if not os.path.exists(image_path):
        return
    with open(image_path, "rb") as f:
        img = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img}");
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("ninja_bg.jpg")

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center; color:#00ffcc;'>🏍️ Bike Price Prediction</h1>", unsafe_allow_html=True)

# ---------------- INPUTS ----------------
company = st.selectbox("Company", ["Honda", "Yamaha", "Bajaj", "TVS", "Kawasaki"])
model_name = st.text_input("Model", "Ninja")
year = st.number_input("Manufactured Year", 2015, 2024, 2021)
warranty = st.number_input("Engine Warranty (Years)", 0, 5, 2)
engine_type = st.selectbox("Engine Type", ["BS4", "BS6"])
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Electric"])
cc = st.number_input("CC", 80, 1000, 300)
fuel_capacity = st.number_input("Fuel Capacity", 5, 25, 15)

# ---------------- PREDICTION ----------------
if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "Company": company,
        "Model": model_name,
        "Manufactured_Year": year,
        "Engine_Warranty": warranty,
        "Engine_Type": engine_type,
        "Fuel_Type": fuel_type,
        "CC": cc,
        "Fuel_Capacity": fuel_capacity
    }])

    input_df = pd.get_dummies(input_df)

    # IMPORTANT: match training columns manually
    trained_cols = model.feature_names_in_
    input_df = input_df.reindex(columns=trained_cols, fill_value=0)

    input_scaled = scaler.transform(input_df)
    price = model.predict(input_scaled)[0]

    st.success(f"Estimated Price: ₹ {price:,.0f}")
