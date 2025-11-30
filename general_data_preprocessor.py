import pandas as pd
import numpy as np
import argparse
import os
from data_cleaning import row_cleaning, column_cleaning, generate_stock_code_list

def standardize_columns_and_replace(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform z-score standardization on specific columns and replace the original columns.
    """
    # Define columns to standardize
    target_columns = ['log_return', 'u_hat_90']
    # Add alpha101 columns (alpha001 to alpha101)
    target_columns.extend([f'alpha{i:03d}' for i in range(1, 172)])
    # Filter columns that actually exist in the dataframe
    cols_to_standardize = [col for col in target_columns if col in data.columns]
    
    if not cols_to_standardize:
        return data

    print(f"Standardizing {len(cols_to_standardize)} columns...")

    # Z-score standardization for each column
    subset = data[cols_to_standardize]
    
    # Calculate Z-scores: (x - mean) / std
    # Handle division by zero if std is 0
    means = subset.mean()
    stds = subset.std()
    
    standardized_subset = (subset - means) / stds
    standardized_subset = standardized_subset.fillna(0)
    
    # Replace original columns with standardized values
    data[cols_to_standardize] = standardized_subset
    
    return data

def preprocess_general_data(input_path: str, output_path: str):
    """
    Clean the data, standardize it, and save to CSV.
    """
    print(f"Reading data from {input_path}...")
    try:
        data = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # 1. Data Cleaning
    print("Starting data cleaning...")
    stock_codes = generate_stock_code_list(data)
    cleaned_data = pd.DataFrame()
    
    for code in stock_codes:
        stock_data = data[data['code'] == code].copy()
        stock_data = row_cleaning(stock_data)
        cleaned_data = pd.concat([cleaned_data, stock_data], ignore_index=True)
    
    cleaned_data = column_cleaning(cleaned_data)
    print("Data cleaning completed.")

    # 2. Standardization (Z-score) and Replacement
    print("Starting standardization...")
    processed_data = standardize_columns_and_replace(cleaned_data)
    print("Standardization completed.")

    # 3. Save to CSV
    print(f"Saving processed data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    processed_data.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General Data Preprocessor: Clean and Standardize")
    parser.add_argument("input_path", type=str, help="Path to the input CSV file")
    parser.add_argument("output_path", type=str, help="Path to the output CSV file")
    
    args = parser.parse_args()

    preprocess_general_data(args.input_path, args.output_path)
