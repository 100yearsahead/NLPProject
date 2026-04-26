from datasets import load_dataset
from statistics import mean
import random


def load_cogs():
    """
    Load the COGS dataset from Hugging Face.

    We fix the dataset source here so the rest of the project
    uses one consistent version of COGS.
    """
    ds = load_dataset("GWHed/cogs")
    return ds


def print_dataset_overview(ds,  n_examples=3, seed=42):
    """
    Print available splits and one example from each split
    so we can inspect the structure of the data.
    """
    random.seed(seed)
    print("Available splits:", list(ds.keys()))

    for split_name in ds.keys():
        split = ds[split_name]
        print(f"\n--- Split: {split_name} ---")
        print(f"Number of examples: {len(split)}")

        # Sample random indices without replacement
        chosen_indices = random.sample(range(len(split)), k=min(n_examples, len(split)))

        for idx in chosen_indices:
            print(f"\nExample {idx}:")
            print(split[idx])


def get_text_lengths(split, source_key, target_key):
    """
    Compute simple whitespace-token lengths for source and target text.
    """
    src_lengths = [len(example[source_key].split()) for example in split]
    tgt_lengths = [len(example[target_key].split()) for example in split]
    return src_lengths, tgt_lengths


def describe_split(split, source_key, target_key):
    """
    Return a dictionary of basic statistics for one split.

    These numbers will go into the report later.
    """
    src_lengths, tgt_lengths = get_text_lengths(split, source_key, target_key)

    return {
            # Number of sentence-logical form pairs in the split
            "num_examples": len(split),

            # Average input sentence length (whitespace token count)
            "avg_src_len": round(mean(src_lengths), 2),

            # Average target logical form length
            "avg_tgt_len": round(mean(tgt_lengths), 2),

            # Longest input sentence in the split
            "max_src_len": max(src_lengths),

            # Longest target logical form in the split
            "max_tgt_len": max(tgt_lengths),

            # Shortest input sentence in the split
            "min_src_len": min(src_lengths),

            # Shortest target logical form in the split
            "min_tgt_len": min(tgt_lengths),
    }


def main():
    ds = load_cogs()
    print_dataset_overview(ds)

    # Update these once you confirm the real column names from the printed example.
    source_key = "source"
    target_key = "target"

    print("\n=== Split statistics ===")
    for split_name in ds.keys():
        stats = describe_split(ds[split_name], source_key, target_key)
        print(f"\n{split_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()