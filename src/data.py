from datasets import load_dataset
from statistics import mean
import random

from vocab import Vocab
import torch
from torch.utils.data import Dataset, DataLoader


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

def build_vocabs(train_split):
    """
    Build separate source and target vocabularies from the training split only.
    """
    src_vocab = Vocab()
    tgt_vocab = Vocab()

    src_vocab.build_from_texts(train_split["source"])
    tgt_vocab.build_from_texts(train_split["target"])

    return src_vocab, tgt_vocab

class COGSDataset(Dataset):
    """
    PyTorch dataset wrapper for one COGS split.

    Each example returns:
    - raw source text
    - raw target text
    - encoded source ids
    - encoded target ids
    """

    def __init__(self, hf_split, src_vocab, tgt_vocab):
        self.examples = hf_split
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]

        src_ids = self.src_vocab.encode(item["source"], add_bos=True, add_eos=True)
        tgt_ids = self.tgt_vocab.encode(item["target"], add_bos=True, add_eos=True)

        return {
            "source_text": item["source"],
            "target_text": item["target"],
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
        }


def collate_fn(batch, src_pad_id, tgt_pad_id):
    """
    Pad variable-length source and target sequences so they can be batched.
    """
    src_seqs = [item["src_ids"] for item in batch]
    tgt_seqs = [item["tgt_ids"] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_seqs, batch_first=True, padding_value=src_pad_id
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_seqs, batch_first=True, padding_value=tgt_pad_id
    )

    return {
        "source_text": [item["source_text"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
        "src_ids": src_padded,
        "tgt_ids": tgt_padded,
    }


def make_dataloaders(ds, batch_size=32):
    """
    Build train/dev/test dataloaders and return them together with the vocabularies.
    """
    train_split = ds["train"]
    dev_split = ds["dev"]
    test_split = ds["test"]

    src_vocab, tgt_vocab = build_vocabs(train_split)

    train_dataset = COGSDataset(train_split, src_vocab, tgt_vocab)
    dev_dataset = COGSDataset(dev_split, src_vocab, tgt_vocab)
    test_dataset = COGSDataset(test_split, src_vocab, tgt_vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, src_vocab.pad_id, tgt_vocab.pad_id),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, src_vocab.pad_id, tgt_vocab.pad_id),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, src_vocab.pad_id, tgt_vocab.pad_id),
    )

    return train_loader, dev_loader, test_loader, src_vocab, tgt_vocab


def main():
    ds = load_cogs()

    # Quick stats summary
    source_key = "source"
    target_key = "target"

    print("\n=== Split statistics ===")
    for split_name in ds.keys():
        stats = describe_split(ds[split_name], source_key, target_key)
        print(f"\n{split_name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    train_loader, dev_loader, test_loader, src_vocab, tgt_vocab = make_dataloaders(ds)

    # Quick sanity check on one batch
    batch = next(iter(train_loader))

    print("\n=== Batch sanity check ===")
    print("src_ids shape:", batch["src_ids"].shape)
    print("tgt_ids shape:", batch["tgt_ids"].shape)
    print("First source text:", batch["source_text"][0])
    print("First target text:", batch["target_text"][0])
    print("Decoded source:", src_vocab.decode(batch["src_ids"][0].tolist()))
    print("Decoded target:", tgt_vocab.decode(batch["tgt_ids"][0].tolist()))
    print("Source vocab size:", len(src_vocab))
    print("Target vocab size:", len(tgt_vocab))


if __name__ == "__main__":
    main()