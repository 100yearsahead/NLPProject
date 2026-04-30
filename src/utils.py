import csv
import os


def sequence_exact_match(pred_tokens, gold_tokens):
    """
    Exact match for one prediction.
    1 if the whole predicted sequence matches the gold sequence, else 0.
    """
    return int(pred_tokens == gold_tokens)


def token_accuracy(pred_tokens, gold_tokens):
    """
    Token-level accuracy for one prediction.

    Compare tokens position by position up to the shorter length.
    This is a simple metric, so it won't capture all structural errors,
    but it is useful alongside exact match.
    """
    if len(gold_tokens) == 0:
        return 0.0

    matched = 0
    compare_len = min(len(pred_tokens), len(gold_tokens))

    for i in range(compare_len):
        if pred_tokens[i] == gold_tokens[i]:
            matched += 1

    return matched / len(gold_tokens)


def append_metrics_row(filepath, row, header=None):
    """
    Append one row of metrics to a CSV file.
    If the file does not exist yet, write the header first.
    """
    file_exists = os.path.exists(filepath)

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header if header else row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)