import pandas as pd

# Load cleaned datasets
historical = pd.read_csv("dataset/cleaned_historical_data.csv")
gps = pd.read_csv("dataset/cleaned_bus_gps_data.csv")
route = pd.read_csv("dataset/cleaned_route_data.csv")

# Show column names
print("===== HISTORICAL DATA COLUMNS =====")
print(historical.columns.tolist())

print("\n===== GPS DATA COLUMNS =====")
print(gps.columns.tolist())

print("\n===== ROUTE DATA COLUMNS =====")
print(route.columns.tolist())

# Show first 5 rows
print("\n===== HISTORICAL DATA =====")
print(historical.head())

print("\n===== GPS DATA =====")
print(gps.head())

print("\n===== ROUTE DATA =====")
print(route.head())