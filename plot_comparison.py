import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comparison():
    # Create output directory
    output_dir = "visualization_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Read data
    if not os.path.exists("comparison_results.csv"):
        print("Error: comparison_results.csv not found.")
        return

    df = pd.read_csv("comparison_results.csv")
    
    # Convert code to string for plotting
    df['code'] = df['code'].astype(str)
    
    # Sort by code for consistent plotting
    df = df.sort_values('code')

    # 1. MSE Comparison Bar Plot
    plt.figure(figsize=(15, 8))
    x = np.arange(len(df))
    width = 0.35
    
    plt.bar(x - width/2, df['deepar_mse'], width, label='DeepAR MSE', color='skyblue')
    plt.bar(x + width/2, df['garch_mse'], width, label='GARCH MSE', color='orange')
    
    plt.xlabel('Stock Code')
    plt.ylabel('MSE')
    plt.title('DeepAR vs GARCH MSE Comparison')
    plt.xticks(x, df['code'], rotation=90, fontsize=8)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_comparison.png"))
    plt.close()

    # 2. MAE Comparison Bar Plot
    plt.figure(figsize=(15, 8))
    
    plt.bar(x - width/2, df['deepar_mae'], width, label='DeepAR MAE', color='lightgreen')
    plt.bar(x + width/2, df['garch_mae'], width, label='GARCH MAE', color='salmon')
    
    plt.xlabel('Stock Code')
    plt.ylabel('MAE')
    plt.title('DeepAR vs GARCH MAE Comparison')
    plt.xticks(x, df['code'], rotation=90, fontsize=8)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_comparison.png"))
    plt.close()

    # 3. MSE Improvement Plot (Positive means DeepAR is better)
    plt.figure(figsize=(15, 8))
    colors = ['green' if x > 0 else 'red' for x in df['mse_improvement']]
    plt.bar(df['code'], df['mse_improvement'], color=colors)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Stock Code')
    plt.ylabel('MSE Improvement (GARCH - DeepAR)')
    plt.title('MSE Improvement (Positive = DeepAR Better)')
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_improvement.png"))
    plt.close()

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    plot_comparison()
