import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# 1. Load the data
df = pd.read_csv("dataset/ml_ready_data.csv")

# 2. For time series, we need to sort by time. 
# We will use 'time_slot' as our time column.
df = df.sort_values(by=["route_id", "time_slot"]).reset_index(drop=True)

# 3. Pick ONE route to start with (let's use route_id = 0)
route_data = df[df["route_id"] == 0]

print("Data for Route 0:")
print(route_data[["time_slot", "avg_delay_minutes"]].head())

# 4. Prepare train and test data
# Let's use 80% for training and 20% for testing
train_size = int(len(route_data) * 0.8)
train_data, test_data = route_data[:train_size], route_data[train_size:]

# 5. Build and train the ARIMA model
print("\nTraining ARIMA model... please wait...")
model = ARIMA(train_data["avg_delay_minutes"], order=(1, 1, 1))
model_fit = model.fit()

# 6. Make predictions
print("Making predictions...")
predictions = model_fit.forecast(steps=len(test_data))

# 7. Check results
mae = mean_absolute_error(test_data["avg_delay_minutes"], predictions)

print("\n--- Time Series Model Results (Route 0) ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")

# 8. Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train_data["time_slot"], train_data["avg_delay_minutes"], label="Training Data")
plt.plot(test_data["time_slot"], test_data["avg_delay_minutes"], label="Actual Delay", color="green")
plt.plot(test_data["time_slot"], predictions, label="Predicted Delay", color="red", linestyle="--")
plt.title("ARIMA Model: Actual vs Predicted Delay (Route 0)")
plt.xlabel("Time Slot")
plt.ylabel("Delay (minutes)")
plt.legend()
plt.tight_layout()
plt.savefig("arima_prediction.png", dpi=150)
plt.show()

print("\nSaved graph as: arima_prediction.png")
