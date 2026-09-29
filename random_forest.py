import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the ML-ready data we made yesterday
df = pd.read_csv("dataset/ml_ready_data.csv")

# 2. Separate inputs (X) and target (y)
# We use these to predict the delay:
X = df[["route_id", "time_slot", "avg_speed"]] 
# This is what we want to predict:
y = df["avg_delay_minutes"]

# 3. Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create the Random Forest model
print("Training Random Forest... please wait...")
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Test the model
y_pred = model.predict(X_test)

# 7. Check how well it did
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Results ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")
print(f"R-squared Score (R2): {r2:.2f}")
