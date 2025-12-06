#!/usr/bin/env python
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============== CONFIG (you can tweak) ==============

ID_COL = "code"
DATE_COL = "date"
RET_COL = "log_return"      # future-return analysis uses this
VOL_COL = "var_true_90"     # volatility label
GARCH_COL = "garch_var_90"  # optional

N_HIST_ALPHAS = 6           # how many alphas to plot hist for
N_TS_STOCKS = 4             # how many stocks for TS plots
ALPHA_FOR_DECILE = "alpha001"
N_DECILES = 10

# ====================================================

def load_data(path: str) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    # parse date
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print("Shape:", df.shape)
    print("Example columns:", df.columns.tolist()[:20])
    return df

def get_alpha_cols(df: pd.DataFrame):
    alpha_cols = [c for c in df.columns if c.startswith("alpha")]
    print(f"Detected {len(alpha_cols)} alpha columns, e.g.:", alpha_cols[:10])
    return alpha_cols

def ensure_out_dir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving figures & CSVs to: {out_dir}/")

# ---------- 1) BASIC STATS ----------

def alpha_basic_stats(df: pd.DataFrame, alpha_cols, out_dir: str):
    stats = pd.DataFrame(index=alpha_cols)
    stats["non_null"] = df[alpha_cols].notna().sum()
    stats["mean"] = df[alpha_cols].mean()
    stats["std"] = df[alpha_cols].std()
    stats["min"] = df[alpha_cols].min()
    stats["max"] = df[alpha_cols].max()

    print("\n=== Alpha basic stats (head) ===")
    print(stats.head())

    stats.to_csv(os.path.join(out_dir, "alpha_basic_stats.csv"))
    print("Saved alpha_basic_stats.csv")

# ---------- 2) HISTOGRAMS ----------

def alpha_histograms(df: pd.DataFrame, alpha_cols, out_dir: str, n_hist: int = N_HIST_ALPHAS):
    chosen = alpha_cols[:n_hist]
    print(f"\nPlotting histograms for alphas: {chosen}")
    n_rows = int(np.ceil(len(chosen) / 3))
    plt.figure(figsize=(12, 3 * n_rows))

    for i, col in enumerate(chosen, 1):
        plt.subplot(n_rows, 3, i)
        df[col].dropna().hist(bins=50)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alpha_histograms.png"))
    plt.close()

# ---------- 3) TIME-SERIES PLOTS ----------

def alpha_time_series(df: pd.DataFrame, alpha_cols, out_dir: str, n_stocks: int = N_TS_STOCKS):
    if ID_COL not in df.columns or DATE_COL not in df.columns:
        print("Skipping time-series plots (no ID/DATE).")
        return

    # pick one alpha for TS plots (e.g. alpha001)
    alpha_col = alpha_cols[0]
    print(f"\nTime-series plots for {alpha_col} on a few stocks.")

    # choose stocks with enough data
    counts = df[ID_COL].value_counts()
    candidates = counts[counts > 50].index.tolist()
    if len(candidates) == 0:
        print("Not enough stocks with >50 rows.")
        return

    rng = np.random.default_rng(42)
    chosen_ids = rng.choice(candidates, size=min(n_stocks, len(candidates)), replace=False)

    plt.figure(figsize=(12, 3 * len(chosen_ids)))
    for i, cid in enumerate(chosen_ids, 1):
        sub = df[df[ID_COL] == cid].sort_values(DATE_COL)
        plt.subplot(len(chosen_ids), 1, i)
        plt.plot(sub[DATE_COL], sub[alpha_col])
        plt.title(f"{alpha_col} for {ID_COL}={cid}")
        plt.xlabel("Date")
        plt.ylabel(alpha_col)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{alpha_col}_time_series_sample_stocks.png"))
    plt.close()

# ---------- 4) CORRELATIONS WITH TARGETS ----------

def alpha_target_correlations(df: pd.DataFrame, alpha_cols, out_dir: str):
    target_cols = []
    for c in [RET_COL, VOL_COL, GARCH_COL]:
        if c in df.columns:
            target_cols.append(c)
    if not target_cols:
        print("No target columns found, skipping correlations.")
        return

    cols = alpha_cols + target_cols
    sub = df[cols].dropna(how="any")
    print(f"\nComputing correlation on shape: {sub.shape}")

    corr = sub.corr(method="pearson")
    corr_alpha_target = corr.loc[alpha_cols, target_cols]

    print("\n=== Alpha–target correlation (head) ===")
    print(corr_alpha_target.head())

    out_csv = os.path.join(out_dir, "alpha_target_correlations.csv")
    corr_alpha_target.to_csv(out_csv)
    print("Saved alpha_target_correlations.csv")

    # heatmap
    plt.figure(figsize=(len(target_cols) * 2.5, 10))
    im = plt.imshow(corr_alpha_target.values, aspect="auto", interpolation="none")
    plt.colorbar(im, label="Pearson corr")
    plt.yticks(range(len(alpha_cols)), alpha_cols, fontsize=6)
    plt.xticks(range(len(target_cols)), target_cols)
    plt.title("Alpha–target correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alpha_target_correlation_heatmap.png"))
    plt.close()

# ---------- 5) DECILE ANALYSIS (ALPHA vs FUTURE RETURN) ----------

def decile_analysis_alpha_vs_future_return(df: pd.DataFrame, out_dir: str, alpha_col: str = ALPHA_FOR_DECILE):
    if alpha_col not in df.columns:
        print(f"Decile analysis skipped: {alpha_col} not in df.")
        return
    if RET_COL not in df.columns or ID_COL not in df.columns or DATE_COL not in df.columns:
        print("Decile analysis skipped: missing ID/DATE/RET.")
        return

    tmp = df[[ID_COL, DATE_COL, alpha_col, RET_COL]].copy()
    tmp = tmp.sort_values([ID_COL, DATE_COL])

    # future return: ret_{t+1}
    tmp["ret_t_plus_1"] = tmp.groupby(ID_COL)[RET_COL].shift(-1)
    tmp = tmp.dropna(subset=[alpha_col, "ret_t_plus_1"])

    # per date: bucket alpha into deciles
    def assign_decile(group):
        try:
            group["alpha_decile"] = pd.qcut(group[alpha_col], N_DECILES,
                                            labels=False, duplicates="drop")
        except ValueError:
            group["alpha_decile"] = np.nan
        return group

    tmp = tmp.groupby(DATE_COL, group_keys=False).apply(assign_decile)
    tmp = tmp.dropna(subset=["alpha_decile"])

    decile_stats = tmp.groupby("alpha_decile")["ret_t_plus_1"].agg(["mean", "std", "count"])
    print(f"\n=== Decile analysis for {alpha_col}: alpha bucket vs next-period return ===")
    print(decile_stats)

    decile_stats.to_csv(os.path.join(out_dir, f"{alpha_col}_decile_vs_future_return.csv"))

    plt.figure()
    plt.bar(decile_stats.index.astype(int), decile_stats["mean"])
    plt.xlabel("alpha_decile (0=lowest, 9=highest)")
    plt.ylabel("Mean next-period return")
    plt.title(f"{alpha_col} deciles vs next-period return")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{alpha_col}_decile_vs_future_return.png"))
    plt.close()

# ---------- MAIN ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Path to ml_dataset_alpha101_volatility.csv")
    parser.add_argument("--out_dir", type=str, default="alpha_eda",
                        help="Directory to save outputs")
    args = parser.parse_args()

    df = load_data(args.data)
    ensure_out_dir(args.out_dir)

    alpha_cols = get_alpha_cols(df)

    # 1) basic stats
    alpha_basic_stats(df, alpha_cols, args.out_dir)

    # 2) histograms
    alpha_histograms(df, alpha_cols, args.out_dir)

    # 3) time series (for first alpha)
    alpha_time_series(df, alpha_cols, args.out_dir)

    # 4) alpha–target correlations
    alpha_target_correlations(df, alpha_cols, args.out_dir)

    # 5) decile analysis for chosen alpha
    decile_analysis_alpha_vs_future_return(df, args.out_dir, alpha_col=ALPHA_FOR_DECILE)

    print("\nDone: alpha EDA complete.")

if __name__ == "__main__":
    main()
