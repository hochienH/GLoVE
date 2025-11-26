# standardize.py: to perform z-score standardization on specific columns of a dataset
import pandas as pd
def standardize(data: pd.DataFrame, column: str):
    mean = data[column].mean()
    std = data[column].std()
    data[column + '_standardized'] = (data[column] - mean) / std
    return data