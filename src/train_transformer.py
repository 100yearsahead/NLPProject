import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from data import load_cogs, make_dataloaders
from models import build_transformer_seq2seq
from utils import sequence_exact_match, token_accuracy, append_metrics_row


def train_one_epoch(model, loader, optimizer, criterion, teacher_forcing_ratio, device):
    """
    Run one training epoch and return average training loss.
    teacher_forcing_ratio is kept here so the function matches the LSTM version,
    even though the Transformer forward pass does not really use it.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)

        optimizer.zero_grad()

        # outputs: [batch_size, tgt_len, tgt_vocab_size]
        outputs = model(src_ids, tgt_ids, teacher_forcing_ratio=teacher_forcing_ratio)

        # skip the first token because that is <bos>
        output_dim = outputs.shape[-1]

        outputs = outputs[:, 1:, :].reshape(-1, output_dim)
        targets = tgt_ids[:, 1:].reshape(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_loss(model, loader, criterion, device):
    """
    Compute average loss on a loader without updating the model.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)

            outputs = model(src_ids, tgt_ids, teacher_forcing_ratio=0.0)

            output_dim = outputs.shape[-1]

            outputs = outputs[:, 1:, :].reshape(-1, output_dim)
            targets = tgt_ids[:, 1:].reshape(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_generation(model, loader, tgt_vocab, device, max_decode_len=50):
    """
    Run greedy decoding and compute:
    - exact match
    - token accuracy
    """
    model.eval()

    exact_matches = []
    token_accuracies = []

    with torch.no_grad():
        for batch in loader:
            src_ids = batch["src_ids"].to(device)
            gold_texts = batch["target_text"]

            decoded_ids = model.greedy_decode(
                src_ids,
                bos_id=tgt_vocab.bos_id,
                eos_id=tgt_vocab.eos_id,
                max_len=max_decode_len,
            )

            for i in range(decoded_ids.shape[0]):
                pred_text = tgt_vocab.decode(decoded_ids[i].tolist())
                gold_text = gold_texts[i]

                pred_tokens = pred_text.split()
                gold_tokens = gold_text.split()

                exact_matches.append(sequence_exact_match(pred_tokens, gold_tokens))
                token_accuracies.append(token_accuracy(pred_tokens, gold_tokens))

    exact_match_score = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
    token_acc_score = sum(token_accuracies) / len(token_accuracies) if token_accuracies else 0.0

    return exact_match_score, token_acc_score


def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Save a checkpoint so we can reload the best model later.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filepath,
    )


def main():
    parser = argparse.ArgumentParser()

    # keep it close to the LSTM script
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.0)
    parser.add_argument("--max_decode_len", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=100)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("../outputs/checkpoints", exist_ok=True)
    os.makedirs("../outputs/tables", exist_ok=True)

    metrics_path = "../outputs/tables/transformer_metrics.csv"
    checkpoint_path = "../outputs/checkpoints/transformer_best.pt"

    # load data
    ds = load_cogs()
    train_loader, dev_loader, test_loader, src_vocab, tgt_vocab = make_dataloaders(
        ds, batch_size=args.batch_size
    )

    # build model
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ignore padding tokens when computing loss
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    best_dev_exact = -1.0

    print("\nStarting transformer training...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            device=device,
        )

        dev_loss = evaluate_loss(model, dev_loader, criterion, device)

        dev_exact, dev_token_acc = evaluate_generation(
            model,
            dev_loader,
            tgt_vocab,
            device,
            max_decode_len=args.max_decode_len,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"dev_loss={dev_loss:.4f} | "
            f"dev_exact={dev_exact:.4f} | "
            f"dev_token_acc={dev_token_acc:.4f}"
        )

        append_metrics_row(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "dev_exact_match": dev_exact,
                "dev_token_accuracy": dev_token_acc,
            },
            header=["epoch", "train_loss", "dev_loss", "dev_exact_match", "dev_token_accuracy"],
        )

        if dev_exact > best_dev_exact:
            best_dev_exact = dev_exact
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            print(f"  saved new best checkpoint to: {checkpoint_path}")

    print("\nTraining finished.")
    print("Best dev exact match:", round(best_dev_exact, 4))


if __name__ == "__main__":
    main()