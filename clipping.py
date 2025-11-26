# to use Gaussian or Student-T distributions, we need to first do log transformation
# to prevent issues with 0 values, we need to add a small constant before log transformation
# but we need to know how small the constant should be
# reed the dataset and find the minimum value
import pandas as pd
import numpy as np

def clipping(input_path: str, clip_value: float):
    data = pd.read_csv(input_path)
    data['var_true_90_clipped'] = data['var_true_90'].clip(lower=clip_value)
    return data

def clip_value_analysis(input_path: str):
    data = pd.read_csv(input_path)
    min_value = data['var_true_90'].min()
    print(f"The minimum value of var_true_90 is: {min_value}")

    data['magnitude'] = data['var_true_90'].apply(lambda x: np.floor(np.log10(x)) if x > 0 else np.nan)
    magnitude_counts = data['magnitude'].value_counts().sort_index()
    print("Distribution of the magnitude of var_true_90:")
    for mag, count in magnitude_counts.items():
        print(f"10^{int(mag)}: {count} values")

def auto_clip_value(input_path: str) -> float:
    # the auto clip value is set if each magnitude count is less than 10 for that magnitude and lower, not cumulative
    data = pd.read_csv(input_path)
    data['magnitude'] = data['var_true_90'].apply(lambda x: np.floor(np.log10(x)) if x > 0 else np.nan)
    magnitude_counts = data['magnitude'].value_counts().sort_index()
    auto_clip = None
    for mag in sorted(magnitude_counts.index):
        if magnitude_counts[mag] < 10:
            auto_clip = 10 ** (mag + 1)
        else:
            break
    print(f"Auto clip value determined: {auto_clip}")
    return auto_clip

if __name__ == "__main__":
    input_path = "Dataset/data/ml_dataset_alpha101_volatility.csv"
    # to analysis datasets magnitude distribution and minimum value, execute the following line
    # clip_value_analysis(input_path)
    auto_clip = auto_clip_value(input_path)
    clipping(input_path, auto_clip)