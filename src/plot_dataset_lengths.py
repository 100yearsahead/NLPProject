import os
import matplotlib.pyplot as plt

from data import load_cogs


# quick helper for counting whitespace-token lengths
def get_lengths(split):
    src_lens = []
    tgt_lens = []

    for ex in split:
        src_lens.append(len(ex["source"].split()))
        tgt_lens.append(len(ex["target"].split()))

    return src_lens, tgt_lens


def main():
    ds = load_cogs()

    # using train split for the dataset analysis figure
    train_data = ds["train"]
    source_lengths, target_lengths = get_lengths(train_data)

    os.makedirs("../outputs/figures", exist_ok=True)

    plt.figure(figsize=(9, 5.5))

    # slightly transparent histograms so overlap is visible
    plt.hist(
        source_lengths,
        bins=range(0, max(target_lengths) + 2),
        alpha=0.75,
        label="Source sentences",
        edgecolor="black",
        linewidth=0.4,
    )

    plt.hist(
        target_lengths,
        bins=range(0, max(target_lengths) + 2),
        alpha=0.55,
        label="Target logical forms",
        edgecolor="black",
        linewidth=0.4,
    )

    # average length markers
    avg_src = sum(source_lengths) / len(source_lengths)
    avg_tgt = sum(target_lengths) / len(target_lengths)

    plt.axvline(avg_src, linestyle="--", linewidth=2, label=f"Avg source = {avg_src:.2f}")
    plt.axvline(avg_tgt, linestyle="--", linewidth=2, label=f"Avg target = {avg_tgt:.2f}")

    plt.title("Source vs Target Length Distribution in COGS", fontsize=14, weight="bold")
    plt.xlabel("Sequence length (whitespace tokens)", fontsize=11)
    plt.ylabel("Number of examples", fontsize=11)

    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_path = "../outputs/figures/source_target_length_distribution.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()