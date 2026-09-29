import pandas as pd

files = [
    "dataset/bus_gps_data.csv",
    "dataset/historical_data.csv",
    "dataset/route_data.csv"
]

for file in files:
    print("\n" + "=" * 60)
    print("FILE:", file)
    print("=" * 60)

    df = pd.read_csv(file)

    print("\nShape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())
    