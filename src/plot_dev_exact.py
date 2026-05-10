from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

lstm_path = ROOT / "outputs" / "tables" / "lstm_FINAL.csv"
transformer_path = ROOT / "outputs" / "tables" / "transformer_metrics.csv"

out_path = ROOT / "outputs" / "figures" / "dev_exact_match_over_epochs.png"


def load_metrics(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find metrics file: {path}")
    return pd.read_csv(path)


def keep_last_run(df):
    # Handles CSVs where multiple runs were accidentally appended.
    run_id = (df["epoch"].diff() < 0).cumsum()
    return df[run_id == run_id.max()].reset_index(drop=True)


def main():
    lstm = keep_last_run(load_metrics(lstm_path))
    transformer = keep_last_run(load_metrics(transformer_path))

    best_lstm_idx = lstm["dev_exact_match"].idxmax()
    best_tr_idx = transformer["dev_exact_match"].idxmax()

    best_lstm_epoch = int(lstm.loc[best_lstm_idx, "epoch"])
    best_lstm_score = float(lstm.loc[best_lstm_idx, "dev_exact_match"])

    best_tr_epoch = int(transformer.loc[best_tr_idx, "epoch"])
    best_tr_score = float(transformer.loc[best_tr_idx, "dev_exact_match"])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        lstm["epoch"],
        lstm["dev_exact_match"],
        marker="o",
        linewidth=2.2,
        label=f"LSTM best = {best_lstm_score:.3f}",
    )

    plt.plot(
        transformer["epoch"],
        transformer["dev_exact_match"],
        marker="o",
        linewidth=2.2,
        label=f"Transformer best = {best_tr_score:.3f}",
    )

    plt.scatter(best_lstm_epoch, best_lstm_score, s=120, edgecolor="black", zorder=5)
    plt.scatter(best_tr_epoch, best_tr_score, s=120, edgecolor="black", zorder=5)

    plt.title("Development Exact Match Over Training", fontsize=14, weight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Development exact match", fontsize=11)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")
    print(f"Best LSTM: epoch={best_lstm_epoch}, dev_exact={best_lstm_score:.4f}")
    print(f"Best Transformer: epoch={best_tr_epoch}, dev_exact={best_tr_score:.4f}")


if __name__ == "__main__":
    main()