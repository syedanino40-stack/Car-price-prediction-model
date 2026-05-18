import streamlit as st
import pickle
import numpy as np

# 1. Load Model and Scaler
with open('car_owner_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('car_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("🚗 Car Owner Type Predictor")
st.write("Enter the car details below to predict its Owner History type:")

# 2. Input Fields based on Dataset columns
year = st.number_input("Manufacturing Year", min_value=2000, max_value=2026, value=2015)
selling_price = st.number_input("Expected Selling Price (in Lakhs)", min_value=0.1, max_value=100.0, value=5.0)
present_price = st.number_input("Current Showroom Present Price (in Lakhs)", min_value=0.1, max_value=100.0, value=8.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# 3. Convert dropdown selections to numbers matching training data
fuel_num = {"Petrol": 0, "Diesel": 1, "CNG": 2}[fuel_type]
seller_num = {"Dealer": 0, "Individual": 1}[seller_type]
trans_num = {"Manual": 0, "Automatic": 1}[transmission]

# 4. Prepare data for model (Array containing exactly 7 features)
features = np.array([[year, selling_price, present_price, kms_driven, fuel_num, seller_num, trans_num]])

# 5. Prediction
if st.button("Predict Owner Type"):
    # Scale inputs first
    scaled_features = scaler.transform(features)
    result = model.predict(scaled_features)
    
    # Display human-readable result based on Owner category
    owner_type = int(result[0])
    if owner_type == 0:
        st.success("✨ **Predicted: First Owner Car** (No previous owners)")
    elif owner_type == 1:
        st.warning("🔄 **Predicted: Second Owner Car**")
    else:
        st.error(f"⚠️ **Predicted: Multiple/Previous Owners Car ({owner_type} Owners)**")
