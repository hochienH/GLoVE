# Repository Guidelines

## Project Structure & Module Organization
- `src/`: active pipeline — `preprocess.py` (feature cleanup), `dataset_builder.py` (TimeSeries splits + static covariates), `model_train.py` (TSMixer + RevIN + custom loss), `model_predict_eval.py` (historical forecasts + metrics).
- `Dataset/`: raw/derived samples and exploratory assets; keep generated CSV/PKL artifacts here.
- `old_src/`: legacy baselines and utilities; read-only reference.
- Outputs: cleaned data, dataset bundles, checkpoints, and evaluation CSVs—store under `Dataset/` or `outputs/`.

## Build, Test, and Development Commands
- Preprocess raw data (disabled features, garch scaling, optional log target):
  ```
  python src/preprocess.py --input Dataset/raw.csv --output Dataset/clean.pkl \
    --disabled_features close log_return u_hat_90 --garch_scaling per_series \
    --use_log_target --target_col target
  ```
- Build train/val/test TimeSeries with static covariates:
  ```
  python src/dataset_builder.py --input Dataset/clean.pkl --output Dataset/ts_data.pkl \
    --val_frac 0.1 --test_frac 0.1 --input_chunk_length 90 --static_mode industry_ticker
  ```
- Train TSMixer (RevIN on, static covariates enabled):
  ```
  python src/model_train.py --data Dataset/ts_data.pkl --epochs 80 --batch_size 32 \
    --lambda 0.6 --log_dir runs --model_path outputs/tsmixer.pth
  ```
- Evaluate vs. GARCH baseline and export metrics:
  ```
  python src/model_predict_eval.py --data Dataset/ts_data.pkl --model outputs/tsmixer.pth \
    --output outputs/metrics.csv --use_log_target
  ```
- Dependencies: Python 3.x, PyTorch, Darts, pandas, numpy. Install via `pip`; use CUDA if available.

## Coding Style & Naming Conventions
- Python modules follow snake_case; keep names descriptive (e.g., `garch_scaling`, `input_chunk_length`). Indent with 4 spaces and prefer type hints on public functions.
- Keep transformations explicit; avoid silent casting outside preprocessing.
- Configuration belongs in CLI flags; update argparse help when adding new options.
- Comments should explain decisions, not restate code; log key hyperparameters when changed.

## Testing Guidelines
- No formal unit tests yet; use `model_predict_eval.py` as the acceptance check after changing preprocessing, splits, or model config.
- If preprocessing used `--use_log_target`, also set it during evaluation to invert predictions.
- Ensure each series meets `input_chunk_length`; short tickers are skipped automatically.
- Compare MAE/RMSE against the GARCH baseline columns; regressions should block merges.

## Commit & Pull Request Guidelines
- Use short imperative commits with prefixes like `feat:`, `fix:`, `refactor:`; keep scope focused.
- PRs should cover purpose, flags/hyperparameters touched, data paths used, and before/after metrics (note if unchanged). Add plots only when evaluating.
- Link related issues/notebooks and note new artifacts (paths, sizes) so reviewers can reproduce quickly.
