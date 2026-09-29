import pandas as pd
from pathlib import Path

# 1. Load the cleaned data
df = pd.read_csv(Path("dataset") / "cleaned_historical_data.csv")

# 2. Drop rows where the target (avg_delay_minutes) is missing
df = df.dropna(subset=["avg_delay_minutes"])

# 3. Fill any missing values in features with 0 (just to be safe)
df = df.fillna(0)

# 4. Save the new ML-ready dataset
output_path = Path("dataset") / "ml_ready_data.csv"
df.to_csv(output_path, index=False)

print("Success! ml_ready_data.csv created.")
print("Final shape of the data:", df.shape)
print("\nColumns saved in ML ready data:")
print(df.columns.tolist())
