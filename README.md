# ADL Final Project - Stock Volatility Prediction with DeepAR

This project implements a DeepAR model for stock volatility prediction and compares its performance against a GARCH baseline.

## Project Structure

- `gluon_data_preprocessor.py`: Preprocesses raw CSV data into GluonTS compatible JSON format.
- `train_deepar.py`: Trains the DeepAR model using the preprocessed data.
- `eval_compare.py`: Evaluates the trained model and compares it with GARCH predictions.
- `plot_comparison.py`: Generates visualization plots for model comparison (MSE/MAE).
- `visualize_prediction.py`: Visualizes individual stock predictions.

## Usage

### 1. Data Preprocessing

First, preprocess the raw dataset. This step cleans the data, handles missing values, and generates JSON files for each stock.

```bash
python3 gluon_data_preprocessor.py ./Dataset/data/ml_dataset_alpha101_volatility.csv --standardize
```

Arguments:
- `input_path`: Path to the input CSV file.
- `--standardize`: (Optional) Apply standardization to features.

### 2. Model Training

Train the DeepAR model using the generated datasets.

```bash
python3 train_deepar.py
```

This script will:
- Load the JSON datasets.
- Train the DeepAR estimator.
- Save the trained model to the `trained_model/` directory.

### 3. Evaluation & Comparison

Evaluate the trained model and compare it against the GARCH baseline present in the original dataset.

```bash
python3 eval_compare.py
```

This will output MSE and MAE metrics for both DeepAR and GARCH, and save the detailed results to `comparison_results.csv`.

### 4. Visualization

#### Model Comparison
To visualize the comparison results (MSE/MAE bar charts and improvement metrics):

```bash
python3 plot_comparison.py
```
The plots will be saved in the `visualization_plots/` directory.

#### Prediction Visualization
To visualize specific prediction results (forecast vs ground truth):

```bash
python3 visualize_prediction.py
```

## Requirements

- Python 3.x
- gluonts==0.16.2
- matplotlib==3.10.7
- numpy==2.3.5
- pandas==2.3.3
- scikit_learn==1.7.2
- torch==2.8.0

or you can just execute
```
pip install -r requirements.txt
```
