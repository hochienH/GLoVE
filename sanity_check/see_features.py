import pickle

with open("Dataset/ts_data.pkl", "rb") as f:
    dataset = pickle.load(f)

print("Features used by model:")
print(dataset["feature_cols"])