import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Load the ML-ready data
df = pd.read_csv("dataset/ml_ready_data.csv")

# 2. Separate inputs (X) and target (y)
X = df[["route_id", "time_slot", "avg_speed"]].values
y = df["avg_delay_minutes"].values

# 3. Reshape X for LSTM: [samples, time_steps, features]
# We will use 1 time step for simplicity
X = X.reshape((X.shape[0], 1, X.shape[1]))

# 4. Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build the LSTM model
print("Building LSTM model...")
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 3))) # 50 neurons, input shape (1 time step, 3 features)
model.add(Dense(1)) # Output layer (1 value: delay)

# 6. Compile the model
model.compile(optimizer='adam', loss='mse')

# 7. Train the model
print("Training LSTM... please wait...")
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# 8. Test the model
y_pred = model.predict(X_test)

# 9. Check how well it did
mae = mean_absolute_error(y_test, y_pred)

print("\n--- LSTM Model Results ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")