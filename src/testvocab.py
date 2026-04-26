from datasets import load_dataset
from vocab import Vocab


def main():
    ds = load_dataset("GWHed/cogs")
    train_split = ds["train"]

    # Build source and target vocabularies from train only
    src_vocab = Vocab()
    tgt_vocab = Vocab()

    src_vocab.build_from_texts(train_split["source"])
    tgt_vocab.build_from_texts(train_split["target"])

    # Check one real example
    example = train_split[0]

    source_text = example["source"]
    target_text = example["target"]

    source_ids = src_vocab.encode(source_text, add_bos=True, add_eos=True)
    target_ids = tgt_vocab.encode(target_text, add_bos=True, add_eos=True)

    print("Original source:")
    print(source_text)
    print("\nEncoded source ids:")
    print(source_ids)
    print("\nDecoded source:")
    print(src_vocab.decode(source_ids))

    print("\n" + "=" * 60)

    print("Original target:")
    print(target_text)
    print("\nEncoded target ids:")
    print(target_ids)
    print("\nDecoded target:")
    print(tgt_vocab.decode(target_ids))

    print("\nSource vocab size:", len(src_vocab))
    print("Target vocab size:", len(tgt_vocab))


if __name__ == "__main__":
    main()