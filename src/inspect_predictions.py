import argparse
import torch
import torch.optim as optim

from data import load_cogs, make_dataloaders
from models import build_lstm_seq2seq


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="../outputs/checkpoints/lstm_best.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_decode_len", type=int, default=50)
    parser.add_argument("--num_examples", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load data
    ds = load_cogs()
    train_loader, dev_loader, test_loader, src_vocab, tgt_vocab = make_dataloaders(
        ds, batch_size=args.batch_size
    )

    # rebuild model with same hyperparameters as training
    model = build_lstm_seq2seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        device=device,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    # dummy optimizer just because checkpoint includes optimizer state
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch = load_checkpoint(model, optimizer, args.checkpoint, device)
    print(f"Loaded checkpoint from epoch {epoch}")

    model.eval()

    printed = 0

    with torch.no_grad():
        for batch in dev_loader:
            src_ids = batch["src_ids"].to(device)
            source_texts = batch["source_text"]
            gold_texts = batch["target_text"]

            decoded_ids = model.greedy_decode(
                src_ids,
                bos_id=tgt_vocab.bos_id,
                eos_id=tgt_vocab.eos_id,
                max_len=args.max_decode_len,
            )

            for i in range(decoded_ids.shape[0]):
                pred_text = tgt_vocab.decode(decoded_ids[i].tolist())

                print("\n" + "=" * 80)
                print(f"Example {printed + 1}")
                print("SOURCE:")
                print(source_texts[i])
                print("\nGOLD:")
                print(gold_texts[i])
                print("\nPRED:")
                print(pred_text)

                printed += 1
                if printed >= args.num_examples:
                    return


if __name__ == "__main__":
    main()