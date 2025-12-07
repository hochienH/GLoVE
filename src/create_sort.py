import csv
import pathlib

LAMBDAS = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
OUTPUT_ROOT = pathlib.Path('outputs')
rows = []

def lam_str(val: float) -> str:
    return str(val).replace('.', '_')

for lam in LAMBDAS:
    lam_s = lam_str(lam)
    t_csv = OUTPUT_ROOT / f'tsmixer_lambda{lam_s}' / 'metrics.csv'
    if t_csv.is_file():
        with t_csv.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    'model_type': 'tsmixer',
                    'lambda': lam,
                    'code': r.get('code'),
                    'MAE_model': r.get('MAE_model'),
                    'RMSE_model': r.get('RMSE_model'),
                    'MAE_garch': r.get('MAE_garch'),
                    'RMSE_garch': r.get('RMSE_garch'),
                })
    l_csv = OUTPUT_ROOT / f'lstm_lambda{lam_s}' / 'metrics.csv'
    if l_csv.is_file():
        with l_csv.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    'model_type': 'lstm',
                    'lambda': lam,
                    'code': r.get('code'),
                    'MAE_model': r.get('MAE_LSTM'),
                    'RMSE_model': r.get('RMSE_LSTM'),
                    'MAE_garch': r.get('MAE_GARCH'),
                    'RMSE_garch': r.get('RMSE_GARCH'),
                })

if not rows:
    raise SystemExit('找不到任何 metrics.csv，請確認 outputs 內的資料。')

out_path = OUTPUT_ROOT / 'all_metrics_combined.csv'
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f'已寫入 {len(rows)} 筆資料到 {out_path}')
