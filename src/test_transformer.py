import torch

from data import load_cogs, make_dataloaders
from models import build_transformer_seq2seq


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    ds = load_cogs()
    train_loader, dev_loader, test_loader, src_vocab, tgt_vocab = make_dataloaders(
        ds, batch_size=8
    )

    # build transformer
    model = build_transformer_seq2seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        device=device,
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        emb_dim=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        max_len=100,
    )

    model.eval()

    # grab one batch
    batch = next(iter(train_loader))

    src_ids = batch["src_ids"].to(device)
    tgt_ids = batch["tgt_ids"].to(device)

    print("src_ids shape:", src_ids.shape)
    print("tgt_ids shape:", tgt_ids.shape)

    # forward pass
    with torch.no_grad():
        outputs = model(src_ids, tgt_ids)

    print("model output shape:", outputs.shape)
    print("expected batch size:", tgt_ids.shape[0])
    print("expected target length:", tgt_ids.shape[1])
    print("target vocab size:", len(tgt_vocab))

    # greedy decode sanity check
    decoded = model.greedy_decode(
        src_ids,
        bos_id=tgt_vocab.bos_id,
        eos_id=tgt_vocab.eos_id,
        max_len=20,
    )

    print("decoded shape:", decoded.shape)

    # inspect one example
    print("\nExample source:")
    print(batch["source_text"][0])

    print("\nGold target:")
    print(batch["target_text"][0])

    print("\nGreedy decoded tokens:")
    print(tgt_vocab.decode(decoded[0].tolist()))


if __name__ == "__main__":
    main()