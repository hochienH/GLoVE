import pandas as pd

df = pd.read_pickle("Dataset/ts_data.pkl")
df.to_csv("Dataset/ts_data.csv", index=False)

