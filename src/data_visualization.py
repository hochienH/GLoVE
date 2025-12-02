import argparse
import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="視覺化模型與 GARCH 的誤差比較。")
    parser.add_argument("--input", required=True, help="輸入 CSV (含 MAE_*, RMSE_* 欄位)。")
    parser.add_argument(
        "--output",
        default=None,
        help="輸出圖檔路徑；未指定時會寫在輸入檔同資料夾下、檔名相同的 .png。",
    )
    return parser.parse_args()


def _resolve_output_path(input_path: pathlib.Path, override: str | None) -> pathlib.Path:
    if override:
        return pathlib.Path(override)
    stem = input_path.with_suffix("").name
    return input_path.parent / f"{stem}.png"


def _group_metric_columns(columns: List[str]) -> Dict[str, Dict[str, str]]:
    grouped: Dict[str, Dict[str, str]] = {}
    for col in columns:
        if "_" not in col:
            continue
        metric, method = col.split("_", 1)
        metric_key = metric.upper()
        grouped.setdefault(metric_key, {})[method.lower()] = col
    return grouped


def _select_pair(metric_name: str, method_dict: Dict[str, str]) -> Tuple[str, str, str]:
    garch_col = next((col for method, col in method_dict.items() if "garch" in method), None)
    model_col = next((col for method, col in method_dict.items() if "model" in method), None)
    if model_col is None:
        model_col = next((col for method, col in method_dict.items() if col != garch_col), None)
    if garch_col is None or model_col is None:
        raise ValueError(f"{metric_name} 缺少可比較的 model/garch 欄位，收到：{list(method_dict.keys())}")
    model_label = model_col.split("_", 1)[1]
    garch_label = garch_col.split("_", 1)[1]
    return model_col, garch_col, (model_label, garch_label)


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = _resolve_output_path(input_path, args.output)

    df = pd.read_csv(input_path)
    if df.shape[1] < 3:
        raise ValueError("CSV 至少需要 3 欄（code + 至少兩個 error 欄位）。")

    metric_groups = _group_metric_columns(df.columns[1:])
    if not metric_groups:
        raise ValueError("找不到類似 MAE_xxx / RMSE_xxx 的欄位命名。請加上 'METRIC_method' 格式。")

    averages = []
    wins_rows = []

    for metric_name, methods in metric_groups.items():
        model_col, garch_col, (model_label, garch_label) = _select_pair(metric_name, methods)

        averages.append(
            {"Metric": metric_name, "Method": model_label, "Average Value": df[model_col].mean()}
        )
        averages.append(
            {"Metric": metric_name, "Method": garch_label, "Average Value": df[garch_col].mean()}
        )

        model_wins = (df[model_col] < df[garch_col]).sum()
        garch_wins = (df[garch_col] < df[model_col]).sum()
        ties = len(df) - model_wins - garch_wins
        wins_rows.append(
            {
                "Metric": metric_name,
                "Model Label": model_label,
                "GARCH Label": garch_label,
                "Model Wins": model_wins,
                "GARCH Wins": garch_wins,
                "Ties": ties,
            }
        )

    average_df = pd.DataFrame(averages)
    plot_data = average_df.pivot(index="Metric", columns="Method", values="Average Value").reset_index()
    wins_df = pd.DataFrame(wins_rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    metrics = plot_data["Metric"].tolist()
    methods = [col for col in plot_data.columns if col not in {"Metric"}]

    bar_width = 0.8 / max(1, len(methods))
    indices = range(len(metrics))

    for idx_method, method in enumerate(methods):
        offsets = [i + idx_method * bar_width for i in indices]
        values = plot_data[method].tolist()
        ax.bar(
            offsets,
            values,
            width=bar_width,
            edgecolor="grey",
            label=method.capitalize(),
        )
        for x, val in zip(offsets, values):
            ax.text(x, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Error Metric", fontweight="bold")
    ax.set_ylabel("Average Error Value", fontweight="bold")
    ax.set_title("Average Error Comparison", fontweight="bold")
    ax.set_xticks([i + bar_width * (len(methods) - 1) / 2 for i in indices])
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print("--- Average Metrics ---")
    print(average_df.to_markdown(index=False))
    print("\n--- Win Counts (Lower error is a win) ---")
    print(wins_df[["Metric", "Model Wins", "GARCH Wins", "Ties"]].to_markdown(index=False))


if __name__ == "__main__":
    main()