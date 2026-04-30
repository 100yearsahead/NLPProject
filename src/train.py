import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from data import load_cogs, make_dataloaders
from models import build_lstm_seq2seq
from utils import sequence_exact_match, token_accuracy, append_metrics_row


def train_one_epoch(model, loader, optimizer, criterion, teacher_forcing_ratio, device):
    """
    Run one training epoch and return average training loss.
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

    These are sequence-level metrics, so they are more meaningful than loss
    once we want to see how good the generated logical forms actually are.
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
    Save a small checkpoint so we can reload the best model later.
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

    # keep the hyperparameters simple for now
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--max_decode_len", type=int, default=50)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("../outputs/checkpoints", exist_ok=True)
    os.makedirs("../outputs/tables", exist_ok=True)

    metrics_path = "../outputs/tables/lstm_metrics.csv"
    checkpoint_path = "../outputs/checkpoints/lstm_best.pt"

    # load data
    ds = load_cogs()
    train_loader, dev_loader, test_loader, src_vocab, tgt_vocab = make_dataloaders(
        ds, batch_size=args.batch_size
    )

    # build model
    model = build_lstm_seq2seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        device=device,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ignore padding tokens when computing loss
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    best_dev_exact = -1.0

    print("\nStarting training...\n")

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

        # save one row per epoch
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

        # save best checkpoint based on dev exact match
        if dev_exact > best_dev_exact:
            best_dev_exact = dev_exact
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            print(f"  saved new best checkpoint to: {checkpoint_path}")

    print("\nTraining finished.")
    print("Best dev exact match:", round(best_dev_exact, 4))


if __name__ == "__main__":
    main()