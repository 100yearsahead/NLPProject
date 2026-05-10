from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# trying to save relative to the project root
ROOT = Path(__file__).resolve().parents[1]
out_path = ROOT / "outputs" / "figures" / "structure_exact_match_bar_chart.png"

# categories for the x-axis
categories = ["Overall", "Passive", "Clausal", "Modifier"]

# final exact match scores from test evaluation
lstm_scores = [0.8397, 0.8473, 0.6712, 0.5732]
transformer_scores = [0.9600, 0.9717, 0.8889, 0.8582]

# x positions
x = np.arange(len(categories))
width = 0.34  # width of each bar

# make figure
fig, ax = plt.subplots(figsize=(9, 5.5))

# bars
bars_lstm = ax.bar(
    x - width / 2,
    lstm_scores,
    width,
    label="LSTM",
    edgecolor="black",
    linewidth=0.8,
)

bars_transformer = ax.bar(
    x + width / 2,
    transformer_scores,
    width,
    label="Transformer",
    edgecolor="black",
    linewidth=0.8,
)

# title and axis labels
ax.set_title("Exact Match by Structure Type on the Test Set", fontsize=14, weight="bold")
ax.set_xlabel("Evaluation subset", fontsize=11)
ax.set_ylabel("Exact match", fontsize=11)

# ticks and limits
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylim(0, 1.05)

# horizontal grid only, makes it cleaner
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

# legend
ax.legend(frameon=True, fontsize=10)

# add score labels above each bar
for bar in bars_lstm:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.015,
        f"{h:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

for bar in bars_transformer:
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.015,
        f"{h:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

# make sure folder exists
out_path.parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {out_path}")