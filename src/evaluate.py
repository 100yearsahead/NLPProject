import os
import argparse
import csv
import torch
import torch.optim as optim

from data import load_cogs, make_dataloaders
from models import build_lstm_seq2seq, build_transformer_seq2seq
from utils import sequence_exact_match, token_accuracy


def load_checkpoint(model, optimizer, checkpoint_path, device):
    # load saved weights + optimizer state
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


def is_passive(source_text):
    # simple rule for passive-ish examples:
    # look for "was/were ... by" in the source sentence
    source = f" {source_text.lower()} "
    return ((" was " in source or " were " in source) and " by " in source)


def is_clausal_complement(target_text):
    # ccomp / xcomp show embedded clause-like structure in the target
    return ("ccomp =" in target_text) or ("xcomp =" in target_text)


def is_modifier_attachment(target_text):
    # nmod markers are a simple way to catch modifier attachment examples
    return "nmod ." in target_text


def build_model(args, src_vocab, tgt_vocab, device):
    # build whichever final model we want to test
    if args.model_type == "lstm":
        model = build_lstm_seq2seq(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            device=device,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    else:
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

    return model


def evaluate_and_save(model, loader, tgt_vocab, device, max_decode_len, output_csv, model_name):
    """
    Run the final model on the full test set.
    Save every prediction to CSV so we can inspect errors later.
    """
    model.eval()

    rows = []
    exact_scores = []
    token_scores = []

    with torch.no_grad():
        for batch in loader:
            src_ids = batch["src_ids"].to(device)
            source_texts = batch["source_text"]
            gold_texts = batch["target_text"]

            # generate full logical forms with greedy decoding
            decoded_ids = model.greedy_decode(
                src_ids,
                bos_id=tgt_vocab.bos_id,
                eos_id=tgt_vocab.eos_id,
                max_len=max_decode_len,
            )

            for i in range(decoded_ids.shape[0]):
                pred_text = tgt_vocab.decode(decoded_ids[i].tolist())
                gold_text = gold_texts[i]
                source_text = source_texts[i]

                pred_tokens = pred_text.split()
                gold_tokens = gold_text.split()

                exact = sequence_exact_match(pred_tokens, gold_tokens)
                tok_acc = token_accuracy(pred_tokens, gold_tokens)

                exact_scores.append(exact)
                token_scores.append(tok_acc)

                # save full row so we can analyse predictions afterwards
                rows.append(
                    {
                        "model": model_name,
                        "source": source_text,
                        "gold_target": gold_text,
                        "pred_target": pred_text,
                        "exact_match": exact,
                        "token_accuracy": tok_acc,
                        "is_passive": int(is_passive(source_text)),
                        "is_ccomp": int(is_clausal_complement(gold_text)),
                        "is_modifier": int(is_modifier_attachment(gold_text)),
                    }
                )

    # make sure the output folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # save all test predictions to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "source",
                "gold_target",
                "pred_target",
                "exact_match",
                "token_accuracy",
                "is_passive",
                "is_ccomp",
                "is_modifier",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    overall_exact = sum(exact_scores) / len(exact_scores) if exact_scores else 0.0
    overall_token = sum(token_scores) / len(token_scores) if token_scores else 0.0

    return overall_exact, overall_token, rows


def subset_score(rows, key):
    # compute exact match on one tagged subset of the test set
    subset = [r for r in rows if r[key] == 1]
    if not subset:
        return 0.0, 0

    score = sum(r["exact_match"] for r in subset) / len(subset)
    return score, len(subset)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, choices=["lstm", "transformer"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_decode_len", type=int, default=40)
    parser.add_argument("--output_csv", type=str, required=True)

    # shared/basic args
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)

    # LSTM args
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)

    # Transformer args
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=100)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # rebuild the dataset and vocabs in the same way as training
    ds = load_cogs()
    train_loader, dev_loader, test_loader, src_vocab, tgt_vocab = make_dataloaders(
        ds, batch_size=args.batch_size
    )

    model = build_model(args, src_vocab, tgt_vocab, device)

    # dummy optimizer just so checkpoint loading works cleanly
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch = load_checkpoint(model, optimizer, args.checkpoint, device)
    print(f"Loaded checkpoint from epoch {epoch}")

    model_name = args.model_type.upper()

    test_exact, test_token, rows = evaluate_and_save(
        model=model,
        loader=test_loader,
        tgt_vocab=tgt_vocab,
        device=device,
        max_decode_len=args.max_decode_len,
        output_csv=args.output_csv,
        model_name=model_name,
    )

    # structure-specific subsets inside the official test set
    passive_score, passive_n = subset_score(rows, "is_passive")
    ccomp_score, ccomp_n = subset_score(rows, "is_ccomp")
    modifier_score, modifier_n = subset_score(rows, "is_modifier")

    print("\n=== Test results ===")
    print(f"overall exact match: {test_exact:.4f}")
    print(f"overall token accuracy: {test_token:.4f}")

    print("\n=== Structure-tagged subsets of test ===")
    print(f"passive exact match: {passive_score:.4f} | n={passive_n}")
    print(f"clausal complement exact match: {ccomp_score:.4f} | n={ccomp_n}")
    print(f"modifier attachment exact match: {modifier_score:.4f} | n={modifier_n}")

    print(f"\nSaved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()