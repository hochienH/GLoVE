#!/usr/bin/env python3
"""
比較 LSTM 與 TSMixer 在各種 lambda 設定下的整體表現。

輸入：eval_all_lambdas.sh 產生的 outputs/all_metrics_combined.csv
輸出：
  1) lambda_vs_metric.png：MAE/RMSE vs. lambda 折線圖（含兩種模型）
  2) best_model_bar.png：最佳 LSTM vs TSMixer 的柱狀圖比較
  3) best_models_summary.csv：最佳結果表格
  4) CLI 印出完整摘要表
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="比較 LSTM 與 TSMixer 在不同 lambda 下的整體表現。"
    )
    parser.add_argument(
        "--input",
        default="outputs/all_metrics_combined.csv",
        help="eval_all_lambdas.sh 產生的彙整 CSV 路徑。",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/comparison",
        help="結果圖表與表格輸出的資料夾。",
    )
    return parser.parse_args()


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "model_type",
        "lambda",
        "code",
        "MAE_model",
        "RMSE_model",
        "MAE_garch",
        "RMSE_garch",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"輸入 CSV 缺少欄位：{sorted(missing)}")
    df["lambda"] = df["lambda"].astype(float)
    df["model_type"] = df["model_type"].str.lower()
    return df


def _aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["model_type", "lambda"], as_index=False)[["MAE_model", "RMSE_model"]]
        .mean()
        .sort_values(["model_type", "lambda"])
    )


def _plot_lambda_curves(agg_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    plt.figure(figsize=(10, 6))
    metrics = ["MAE_model", "RMSE_model"]
    titles = ["MAE vs. Lambda", "RMSE vs. Lambda"]

    for idx, metric in enumerate(metrics, start=1):
        plt.subplot(2, 1, idx)
        for model_type, group_df in agg_df.groupby("model_type"):
            plt.plot(
                group_df["lambda"],
                group_df[metric],
                marker="o",
                label=model_type.upper(),
            )
        plt.title(titles[idx - 1], fontweight="bold")
        plt.xlabel("Lambda")
        plt.ylabel(metric)
        plt.grid(alpha=0.3)
        if idx == 1:
            plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _find_best_per_model(agg_df: pd.DataFrame) -> pd.DataFrame:
    best_rows = []
    for model_type, group_df in agg_df.groupby("model_type"):
        best_mae_row = group_df.loc[group_df["MAE_model"].idxmin()]
        best_rmse_row = group_df.loc[group_df["RMSE_model"].idxmin()]
        best_rows.append(
            {
                "model_type": model_type,
                "best_lambda_mae": best_mae_row["lambda"],
                "best_mae": best_mae_row["MAE_model"],
                "best_lambda_rmse": best_rmse_row["lambda"],
                "best_rmse": best_rmse_row["RMSE_model"],
            }
        )
    return pd.DataFrame(best_rows)


def _plot_best_bar(best_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    labels = best_df["model_type"].str.upper().tolist()
    mae_vals = best_df["best_mae"].tolist()
    rmse_vals = best_df["best_rmse"].tolist()

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar([i - width / 2 for i in x], mae_vals, width=width, label="MAE", color="#4C72B0")
    plt.bar([i + width / 2 for i in x], rmse_vals, width=width, label="RMSE", color="#55A868")

    for idx, val in enumerate(mae_vals):
        plt.text(idx - width / 2, val + 0.005, f"{val:.4f}", ha="center", va="bottom")
    for idx, val in enumerate(rmse_vals):
        plt.text(idx + width / 2, val + 0.005, f"{val:.4f}", ha="center", va="bottom")

    plt.xticks(list(x), labels)
    plt.ylabel("Error")
    plt.title("最佳 LSTM vs TSMixer 表現比較", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _print_summary(best_df: pd.DataFrame) -> None:
    print("\n=== 最佳結果摘要（依 MAE / RMSE） ===")
    print(best_df.to_markdown(index=False, floatfmt=".6f"))
    print("")


def _export_summary(best_df: pd.DataFrame, output_path: pathlib.Path) -> None:
    best_df.to_csv(output_path, index=False)
    print(f"最佳結果表格已輸出至：{output_path}")


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output_dir)

    if not input_path.is_file():
        raise FileNotFoundError(f"找不到輸入檔：{input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = _ensure_columns(df)
    agg_df = _aggregate_metrics(df)

    lambda_plot = output_dir / "lambda_vs_metric.png"
    _plot_lambda_curves(agg_df, lambda_plot)
    print(f"Lambda 折線圖輸出至：{lambda_plot}")

    best_df = _find_best_per_model(agg_df)
    best_plot = output_dir / "best_model_bar.png"
    _plot_best_bar(best_df, best_plot)
    print(f"最佳模型比較圖輸出至：{best_plot}")

    summary_csv = output_dir / "best_models_summary.csv"
    _export_summary(best_df, summary_csv)

    _print_summary(best_df)


if __name__ == "__main__":
    main()
