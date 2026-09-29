import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page settings
st.set_page_config(
    page_title="Public Transport Delay Prediction",
    page_icon="🚍"
)

st.title("🚍 Real-Time Public Transport Delay Prediction")
st.write("Predicting bus delays using GPS and historical data")

# Load historical data
historical = pd.read_csv("dataset/historical_data.csv")

# Prepare training data
X = historical[["route_id", "time_slot", "avg_speed"]]
y = historical["avg_delay_minutes"]

# Train model
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
rmse = mean_squared_error(y, y_pred) ** 0.5

st.write("MAE:", mae)
st.write("RMSE:", rmse)

# Load GPS data
gps = pd.read_csv("dataset/bus_gps_data.csv")

# Convert timestamp
gps["timestamp"] = pd.to_datetime(gps["timestamp"])
gps["time_slot"] = gps["timestamp"].dt.hour

# Predict delays
gps_features = gps[
    ["route_id", "time_slot", "speed"]
].copy()

gps_features = gps_features.rename(
    columns={"speed": "avg_speed"}
)

gps["predicted_delay_minutes"] = model.predict(gps_features)

# Select a bus
bus_id = st.selectbox(
    "Select Bus ID",
    gps["bus_id"].unique()
)

# Get selected bus
selected_bus = gps[gps["bus_id"] == bus_id].iloc[0]

# Display information
st.subheader("🚌 Current Bus Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Bus ID", selected_bus["bus_id"])

with col2:
    st.metric("Route ID", selected_bus["route_id"])

with col3:
    st.metric("Speed", f"{selected_bus['speed']} km/h")

st.subheader("⏱️ Predicted Delay")

delay = selected_bus["predicted_delay_minutes"]

st.metric(
    "Expected Delay",
    f"{delay:.2f} minutes"
)

# Status
if delay < 3:
    st.success("🟢 Bus is approximately on time")
elif delay < 6:
    st.warning("🟡 Bus may be slightly delayed")
else:
    st.error("🔴 Bus is significantly delayed")

# Location
st.subheader("📍 Bus Location")

st.write(
    f"Latitude: {selected_bus['latitude']}"
)

st.write(
    f"Longitude: {selected_bus['longitude']}"
)

st.subheader("📊 GPS Data")

st.dataframe(
    gps[
        [
            "bus_id",
            "route_id",
            "timestamp",
            "latitude",
            "longitude",
            "speed",
            "predicted_delay_minutes"
        ]
    ]
)
