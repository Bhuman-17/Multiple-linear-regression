import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("Multiple Linear Regression model_github.joblib")

# Define input feature names
features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
            'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']

st.title("🏠 Boston Housing Price Predictor")
st.write("Enter the values for each feature below to predict the median house price (`medv`).")

# Collect user inputs
user_input = {}
for feature in features:
    if feature in ['chas', 'rad', 'tax']:  # likely to be integer-based
        user_input[feature] = st.number_input(f"{feature}", step=1, value=1)
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0.0, format="%.2f")

# Convert inputs to DataFrame
input_df = pd.DataFrame([user_input])

# Predict when button clicked
if st.button("Predict MEDV"):
    prediction = model.predict(input_df)
    st.success(f"🏡 Predicted MEDV: ₹{prediction[0]:.2f}")
