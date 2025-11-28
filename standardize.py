import pandas as pd
import numpy as np

def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform z-score standardization on specific columns.
    """
    # Define columns to standardize
    target_columns = ['log_return', 'u_hat_90']
    # Add alpha101 columns (alpha001 to alpha101)
    target_columns.extend([f'alpha{i:03d}' for i in range(1, 102)])
    # Filter columns that actually exist in the dataframe
    cols_to_standardize = [col for col in target_columns if col in data.columns]
    
    if not cols_to_standardize:
        return data

    # Z-score standardization for each column
    # Using vectorized operation for efficiency
    subset = data[cols_to_standardize]
    # if std is zero, then the standardized value will be NaN, handle that case
    standardized_subset = (subset - subset.mean()) / subset.std()
    standardized_subset = standardized_subset.fillna(0)
    
    # Rename columns to have '_standardized' suffix
    standardized_subset.columns = [f"{col}_standardized" for col in cols_to_standardize]
    
    # Concatenate with original data
    data = pd.concat([data, standardized_subset], axis=1)
    
    return data

if __name__ == "__main__":
    input_path = 'Dataset/data/ml_dataset_alpha101_volatility_cleaned.csv'
    output_path = 'Dataset/data/ml_dataset_alpha101_volatility_standardized.csv'
    try:
        dataset = pd.read_csv(input_path)
        processed_dataset = standardize_columns(dataset)
        processed_dataset.to_csv(output_path, index=False)
        
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")