import pandas as pd

df = pd.read_csv("real_gas_storage_data.csv")
print("Start Date:", df["Date"].min())
print("End Date:", df["Date"].max())
print("Total Daily Records:", len(df))
print("\nDataset Preview:\n", df.head())