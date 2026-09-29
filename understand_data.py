import pandas as pd

files = [
    "dataset/cleaned_historical_data.csv",
    "dataset/cleaned_bus_gps_data.csv",
    "dataset/cleaned_route_data.csv"
]

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", file)
    print("=" * 60)

    df = pd.read_csv(file)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nBasic Statistics:")
    print(df.describe())

print("\n========================================")
print("DATA UNDERSTANDING COMPLETED")
print("========================================")
