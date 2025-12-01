# dataset_generator.py: generate datasets for gluonTS
import os
import numpy as np
import pandas as pd
import json
import argparse
from standardize import standardize_columns

# first check stock codes we want to use
def generate_stock_code_list(data: pd.DataFrame) -> list:
    stock_codes = data['code'].unique().tolist()
    return stock_codes

def generate_datasets_per_stock(data: pd.DataFrame, stock_code: int, standardize: bool = False) -> json:
    datasets = json.dumps({})
    stock_data = data[data['code'] == stock_code].copy()


    # create a time series dataset for gluonTS
    time_series = {
        "start": str(stock_data['date'].min())
    }
    
    # for the target values, we use 'var_true_90' column
    time_series["target"] = stock_data['var_true_90'].tolist()

    # static covariates is the stock_data columns that only have one unique value
    DICT_DIR = "gluonTS_Dataset/static_dict"
    static_covariates = []
    potential_static_cols = sorted(stock_data.columns)

    for col in potential_static_cols:
        # skip alpha features
        if col.startswith('alpha'):
            continue
        
        # Allow industry_code to be static even if it changes (take the latest)
        if stock_data[col].nunique() == 1 or col == 'industry_code':
            # get the last value (most recent)
            raw_val = stock_data[col].iloc[-1]
            if isinstance(raw_val, (np.integer, np.floating)):
                raw_val = raw_val.item()

            #define key
            str_key = str(raw_val)

            # define the path
            dict_path = os.path.join(DICT_DIR, f"{col}.json")

            # read or create the dictionary file
            if os.path.exists(dict_path):
                with open(dict_path, 'r', encoding='utf-8') as f:
                    try:
                        mapping_dict = json.load(f)
                    except json.JSONDecodeError:
                        mapping_dict = {} # 防止檔案損毀
            else:
                mapping_dict = {}

            # encoding
            if str_key in mapping_dict:
                encoded_val = mapping_dict[str_key]
            else:
                encoded_val = len(mapping_dict)
                mapping_dict[str_key] = encoded_val
                with open(dict_path, 'w', encoding='utf-8') as f:
                    json.dump(mapping_dict, f, indent=4, ensure_ascii=False)
            
            static_covariates.append(encoded_val)
    time_series["feat_static_cat"] = static_covariates

    # dynamic covariates consist two parts:
    dynamic_covariates = {}

    # the first part is general features
    general_features = ['close', 'garch_var_90']
    for feature in general_features:
            if feature in stock_data.columns:
                dynamic_covariates[feature] = stock_data[feature].tolist()

    # the second part is other garch features (may be standardize first)
    garch_features = ['log_return','u_hat_90']

    # the third part is the alpha101 features, we only give the name rule, model needs to iterate them
    alpha_features = [f'alpha{i:03d}' for i in range(1, 102)]

    other_features = garch_features + alpha_features
    
    dynamic_covariates = []
    if standardize:
        # use the standarize version of the features to replace the second and the third parts, after standardize, those columns will have '_standardized' suffix
        stock_data = standardize_columns(stock_data)
        for feature in other_features:
            standardized_feature = f"{feature}_standardized"
            if standardized_feature in stock_data.columns:
                dynamic_covariates.append(stock_data[standardized_feature].tolist())
    else:
        for feature in other_features:
            if feature in stock_data.columns:
                dynamic_covariates.append(stock_data[feature].tolist())
                
    time_series["feat_dynamic_real"] = dynamic_covariates

    datasets = json.dumps({
        "start": time_series["start"],
        "target": time_series["target"],
        "feat_static_cat": time_series["feat_static_cat"],
        "feat_dynamic_real": time_series["feat_dynamic_real"]
    })
    return datasets

# test for code == 2330
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate datasets for GluonTS.')
    parser.add_argument('--standardize', action='store_true', help='Standardize features')
    args = parser.parse_args()

    # check if the output directory exists
    os.makedirs("gluonTS_Dataset/input", exist_ok=True)
    os.makedirs("gluonTS_Dataset/static_dict", exist_ok=True)

    input_path = "Dataset/data/ml_dataset_alpha101_volatility_cleaned.csv"
    data = pd.read_csv(input_path)
    stock_code_list = generate_stock_code_list(data)
    for code in stock_code_list:
        datasets_json = generate_datasets_per_stock(data, code, standardize=args.standardize)
        # json dump to file
        output_path = f"gluonTS_Dataset/input/dataset_{code}.json"
        with open(output_path, 'w') as f:
            f.write(datasets_json)