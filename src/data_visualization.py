import pandas as pd
import io
import matplotlib.pyplot as plt
import argparse
import pathlib
import tabulate
# Data provided by the user
# 說明自己的資料來源




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Darts TimeSeries datasets for TSMixer.")
    parser.add_argument("--input", required=True, help="Path to cleaned data (CSV).")
    parser.add_argument("--output", required=False, help="Path to save the plot.", default="output/average_error_comparison.png")
    return parser.parse_args()
def main():
    args = parse_args()
    input_path = pathlib.Path(args.input)
    input_name = input_path.name.split('.csv')[0]
    if args.output is None:
        output_path = pathlib.Path(args.output)
    else:
        output_path = pathlib.Path(args.input).parent / f"{input_name}.png"
    df = pd.read_csv(input_path)

    # --- 1. Calculate Averages ---
    average_metrics = df[['MAE_model', 'RMSE_model', 'MAE_garch', 'RMSE_garch']].mean().to_frame(name='Average Value')

    # --- 2. Calculate Wins (A "win" means lower error) ---

    # Model wins for MAE
    mae_model_wins = (df['MAE_model'] < df['MAE_garch']).sum()
    # GARCH wins for MAE
    mae_garch_wins = (df['MAE_garch'] < df['MAE_model']).sum()
    # Ties for MAE
    mae_ties = len(df) - mae_model_wins - mae_garch_wins

    # Model wins for RMSE
    rmse_model_wins = (df['RMSE_model'] < df['RMSE_garch']).sum()
    # GARCH wins for RMSE
    rmse_garch_wins = (df['RMSE_garch'] < df['RMSE_model']).sum()
    # Ties for RMSE
    rmse_ties = len(df) - rmse_model_wins - rmse_garch_wins

    # Consolidate win counts
    win_counts = pd.DataFrame({
        'Metric': ['MAE', 'RMSE'],
        'Model Wins': [mae_model_wins, rmse_model_wins],
        'GARCH Wins': [mae_garch_wins, rmse_garch_wins],
        'Ties': [mae_ties, rmse_ties]
    })

    # --- 3. Prepare Data for Plotting (Averages) ---
    plot_data = average_metrics.reset_index()
    plot_data[['Metric', 'Method']] = plot_data['index'].str.split('_', expand=True).iloc[:, [0, 1]]
    plot_data = plot_data.pivot(index='Metric', columns='Method', values='Average Value').reset_index()

    # --- 4. Generate Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    metrics = plot_data['Metric']
    model_means = plot_data['model']
    garch_means = plot_data['garch']

    bar_width = 0.35
    r1 = range(len(metrics))
    r2 = [x + bar_width for x in r1]

    ax.bar(r1, model_means, color='skyblue', width=bar_width, edgecolor='grey', label='Model')
    ax.bar(r2, garch_means, color='lightcoral', width=bar_width, edgecolor='grey', label='GARCH')

    # Add labels for bars
    for i in r1:
        ax.text(r1[i], model_means[i] + 0.01, f'{model_means[i]:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(r2[i], garch_means[i] + 0.01, f'{garch_means[i]:.3f}', ha='center', va='bottom', fontsize=9)

    # Add titles and labels
    ax.set_xlabel('Error Metric', fontweight='bold')
    ax.set_ylabel('Average Error Value', fontweight='bold')
    ax.set_title('Average MAE and RMSE Comparison: Model vs GARCH', fontweight='bold')
    ax.set_xticks([r + bar_width/2 for r in range(len(metrics))])
    ax.set_xticklabels(metrics)

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print("--- Average Metrics ---")
    print(average_metrics.to_markdown())
    print("\n--- Win Counts (Lower error is a win) ---")
    print(win_counts.to_markdown(index=False))


if __name__ == "__main__":
    main()