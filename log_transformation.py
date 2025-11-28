# log_transformation.py: to perform log transformation on the clipped dataset
import pandas as pd
import numpy as np

def log_transformation(data: pd.DataFrame) -> pd.DataFrame:
    # perform log transformation on 'var_true_90_clipped'
    data['var_true_90_log'] = data['var_true_90_clipped'].apply(lambda x: np.log(x))
    return data

if __name__ == "__main__":
    input_path = "Dataset/data/ml_dataset_alpha101_volatility_clipped.csv"
    output_path = "Dataset/data/ml_dataset_alpha101_volatility_log_transformed.csv"
    data = pd.read_csv(input_path)
    transformed_data = log_transformation(data)
    # save the log-transformed dataset
    # transformed_data.to_csv(output_path, index=False)