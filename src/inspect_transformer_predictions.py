import argparse
import torch
import torch.optim as optim

from data import load_cogs, make_dataloaders
from models import build_transformer_seq2seq


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="../outputs/checkpoints/transformer_best.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max_decode_len", type=int, default=40)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--num_examples", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds = load_cogs()
    train_loader, dev_loader, test_loader, src_vocab, tgt_vocab = make_dataloaders(
        ds, batch_size=args.batch_size
    )

    model = build_transformer_seq2seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        device=device,
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        emb_dim=args.emb_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

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
                gold_text = gold_texts[i]
                exact = int(pred_text.split() == gold_text.split())

                print("\n" + "=" * 80)
                print(f"Example {printed + 1}")
                print("EXACT MATCH:", exact)
                print("SOURCE:")
                print(source_texts[i])
                print("\nGOLD:")
                print(gold_text)
                print("\nPRED:")
                print(pred_text)

                printed += 1
                if printed >= args.num_examples:
                    return


if __name__ == "__main__":
    main()