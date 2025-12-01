# data_cleaning.py: functions for cleaning dataset
import pandas as pd
import numpy as np

def generate_stock_code_list(data: pd.DataFrame) -> list:
    stock_codes = data['code'].unique().tolist()
    return stock_codes

def row_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    # i want to drop rows from the start and end of the dataframe if they contain any NaN value
    # for i from the first row to the last row, if there is no NaN 
    
    # Reset index to ensure iloc works as expected if indices are not continuous
    data = data.reset_index(drop=True)
    
    start = 0
    end = len(data) - 1
    counter = 0
    
    # Find the first valid index
    while start <= end:
        if data.iloc[start].isna().any():
            start += 1
            counter += 1
        else:
            break
            
    # Find the last valid index
    while end >= start:
        if data.iloc[end].isna().any():
            end -= 1
            counter += 1
        else:
            break
            
    # Slice the dataframe to keep only the valid range
    if start <= end:
        data = data.iloc[start : end + 1].copy()
    else:
        # If all rows were dropped
        data = data.iloc[0:0].copy()
        
    print(f"Dropped {counter} rows from start and end containing NaN values.")
    return data

def column_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    # consider columns, if the column has any NaN value or inf value, drop that column
    cols_to_drop = []
    for col in data.columns:
        # Check for NaN
        if data[col].isna().any():
            cols_to_drop.append(col)
            continue
            
        # Check for Inf (only for numeric columns)
        if pd.api.types.is_numeric_dtype(data[col]):
            if np.isinf(data[col]).any():
                cols_to_drop.append(col)
    print(f"Dropping columns: {cols_to_drop}")
    data = data.drop(columns=cols_to_drop)
    return data

if __name__ == "__main__":
    # example usage
    input_path = "Dataset/data/ml_dataset_alpha101_volatility.csv"
    data = pd.read_csv(input_path)
    stock_codes = generate_stock_code_list(data)
    cleaned_data = pd.DataFrame()
    for code in stock_codes:
        stock_data = data[data['code'] == code].copy()
        stock_data = row_cleaning(stock_data)
        cleaned_data = pd.concat([cleaned_data, stock_data], ignore_index=True)
    cleaned_data = column_cleaning(cleaned_data)
    output_path = "Dataset/data/ml_dataset_alpha101_volatility_cleaned.csv"
    cleaned_data.to_csv(output_path, index=False)
    # print common features across all stocks
    common_features = set(cleaned_data.columns)
    print("Common features across all stocks:")
    for feature in common_features:
        print(feature)
        