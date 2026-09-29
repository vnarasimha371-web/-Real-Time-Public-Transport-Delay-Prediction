import pandas as pd

# Load datasets
historical = pd.read_csv("dataset/historical_data.csv")
gps = pd.read_csv("dataset/bus_gps_data.csv")
route = pd.read_csv("dataset/route_data.csv")

# Remove duplicate rows
historical = historical.drop_duplicates()
gps = gps.drop_duplicates()
route = route.drop_duplicates()

# Display missing values after cleaning
print("Historical data missing values:")
print(historical.isnull().sum())

print("\nGPS data missing values:")
print(gps.isnull().sum())

print("\nRoute data missing values:")
print(route.isnull().sum())

# Save cleaned datasets
historical.to_csv("dataset/cleaned_historical_data.csv", index=False)
gps.to_csv("dataset/cleaned_bus_gps_data.csv", index=False)
route.to_csv("dataset/cleaned_route_data.csv", index=False)

print("\nCleaning completed successfully!")