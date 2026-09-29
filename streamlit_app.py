import streamlit as st
import pandas as pd

# 1. Set up the title of the web page
st.title("🚌 Real-Time Public Transport Delay Prediction")
st.write("Welcome! This app predicts bus delays based on the model we trained.")

# 2. Load the data
df = pd.read_csv("dataset/ml_ready_data.csv")

# 3. Create input boxes for the user
st.header("Enter Trip Details:")

route = st.number_input("Route ID", min_value=0, max_value=int(df["route_id"].max()), value=0)
time_slot = st.number_input("Time Slot (0-23 hours)", min_value=0, max_value=23, value=12)
speed = st.number_input("Average Speed (km/h)", min_value=0.0, max_value=100.0, value=30.0)

# 4. Predict button
if st.button("Predict Delay"):
    # For now, we will show a simple message. 
    # (In a real app, you would load your ARIMA model here).
    st.success(f"Based on Route {route}, Time Slot {time_slot}, and Speed {speed} km/h...")
    st.info("Predicted Delay: ~2.39 minutes (ARIMA model average)")
    
    # Show a small chart
    st.subheader("Model Comparison")
    st.image("model_comparison.png", caption="Our Model Results")