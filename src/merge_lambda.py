import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge metrics across different lambda_weights and plot MAE/RMSE vs lambda."
    )
    parser.add_argument(
        "--base_dir",
        default="outputs/Base",
        help="根目錄，底下應該有 Lambda_* 子資料夾。",
    )
    parser.add_argument(
        "--output_csv",
        default="outputs/Base/lambda_summary.csv",
        help="合併後的 summary CSV 輸出路徑。",
    )
    parser.add_argument(
        "--output_fig",
        default="outputs/Base/lambda_metrics.png",
        help="MSE/MAE 走勢圖輸出路徑。",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    rows = []

    # 掃所有 Lambda_* 資料夾
    for subdir in sorted(base_dir.glob("Lambda_*")):
        if not subdir.is_dir():
            continue

        name = subdir.name  # e.g., "Lambda_0p01"
        if "Lambda_" not in name:
            continue
        lam_str = name.split("Lambda_")[1]
        try:
            lam_val = float(lam_str.replace("p", "."))
        except ValueError:
            print(f"Warning: 無法從資料夾名稱解析 lambda: {name}")
            continue

        metrics_path = subdir / "metrics.csv"
        if not metrics_path.exists():
            print(f"Warning: {metrics_path} 不存在，略過")
            continue

        df = pd.read_csv(metrics_path)
        if df.empty:
            print(f"Warning: {metrics_path} 為空，略過")
            continue

        # 取所有股票的平均指標
        mae_model = df["MAE_model"].mean()
        rmse_model = df["RMSE_model"].mean()
        mae_garch = df["MAE_garch"].mean()
        rmse_garch = df["RMSE_garch"].mean()
        qlike_model = df["QLIKE_model"].mean() if "QLIKE_model" in df.columns else np.nan
        qlike_garch = df["QLIKE_garch"].mean() if "QLIKE_garch" in df.columns else np.nan

        rows.append(
            {
                "lambda_weight": lam_val,
                "MAE_model": mae_model,
                "RMSE_model": rmse_model,
                "MAE_garch": mae_garch,
                "RMSE_garch": rmse_garch,
                "QLIKE_model": qlike_model,
                "QLIKE_garch": qlike_garch,
            }
        )

    if not rows:
        print("沒有找到任何 Lambda_* metrics，可以確認資料夾與檔案是否存在。")
        return

    summary = pd.DataFrame(rows).sort_values("lambda_weight")
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"合併結果已輸出到 {out_csv}")

    # 畫圖：RMSE / MAE / QLIKE，model 與 GARCH 一起
    lambdas = summary["lambda_weight"].values
    rmse_m = summary["RMSE_model"].values
    rmse_g = summary["RMSE_garch"].values
    mae_m = summary["MAE_model"].values
    mae_g = summary["MAE_garch"].values
    ql_m = summary["QLIKE_model"].values
    ql_g = summary["QLIKE_garch"].values

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    ax1, ax2, ax3 = axes

    # RMSE
    ax1.plot(lambdas, rmse_m, marker="o", label="RMSE_model")
    ax1.plot(lambdas, rmse_g, marker="s", label="RMSE_garch")
    ax1.set_ylabel("RMSE")
    ax1.set_title("TSMixer vs GARCH across lambda_weight")
    ax1.legend()

    # MAE
    ax2.plot(lambdas, mae_m, marker="o", label="MAE_model")
    ax2.plot(lambdas, mae_g, marker="s", label="MAE_garch")
    ax2.set_ylabel("MAE")
    ax2.legend()

    # QLIKE（越小越好）
    ax3.plot(lambdas, ql_m, marker="o", label="QLIKE_model")
    ax3.plot(lambdas, ql_g, marker="s", label="QLIKE_garch")
    ax3.set_xlabel("lambda_weight")
    ax3.set_ylabel("QLIKE")
    ax3.legend()

    plt.tight_layout()
    out_fig = Path(args.output_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"圖已輸出到 {out_fig}")


if __name__ == "__main__":
    main()
