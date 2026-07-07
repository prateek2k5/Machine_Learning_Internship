import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏡 California Housing Price Prediction")

st.write("Enter the housing details below:")

MedInc = st.number_input("Median Income")
HouseAge = st.number_input("House Age")
AveRooms = st.number_input("Average Rooms")
AveBedrms = st.number_input("Average Bedrooms")
Population = st.number_input("Population")
AveOccup = st.number_input("Average Occupancy")
Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

if st.button("Predict Price"):

    data = np.array([[

        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude

    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(f"Predicted House Value: {prediction[0]:.2f}")