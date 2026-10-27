"""View-specific fake-news probabilities and binary metrics."""

import torch

from .data import class_token_ids, to_device
from .runtime import prepared_batches


def binary_metrics(targets, scores, threshold=0.5):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    if not targets:
        return {"n": 0, "accuracy": None, "macro_f1": None, "auroc": None}
    preds = [int(score >= threshold) for score in scores]
    if len(targets) != len(preds) or any(y not in [0, 1] for y in targets):
        raise ValueError("Metrics require matched binary labels and scores.")
    both = set(targets) == {0, 1}
    real = [i for i, y in enumerate(targets) if y == 0]
    fake = [i for i, y in enumerate(targets) if y == 1]
    return {
        "n": len(targets),
        "accuracy": float(accuracy_score(targets, preds)) if both else None,
        "macro_f1": float(f1_score(targets, preds, labels=[0, 1], average="macro", zero_division=0)) if both else None,
        "auroc": float(roc_auc_score(targets, scores)) if both else None,
        "real_recall": sum(preds[i] == 0 for i in real) / len(real) if real else None,
        "false_positive_rate": sum(preds[i] == 1 for i in real) / len(real) if real else None,
        "fake_recall": sum(preds[i] == 1 for i in fake) / len(fake) if fake else None,
        "false_negative_rate": sum(preds[i] == 0 for i in fake) / len(fake) if fake else None,
        "class_counts": {"real": len(real), "fake": len(fake)},
    }


def predict(model, rows, collator, processor, size, device):
    model.eval()
    result = []
    ids = class_token_ids(processor)
    with torch.inference_mode():
        for _, item in prepared_batches(rows, collator, size):
            logits = model.evaluate_batch(to_device(item, device)["batch"])
            probabilities = logits[:, ids].softmax(-1)[:, 1].cpu().tolist()
            result.extend({"id": row["id"], "label": row["label"], "fake_probability": probability}
                          for row, probability in zip(item["rows"], probabilities))
    if len(result) != len(rows):
        raise RuntimeError("Prediction count mismatch.")
    return result


def summarize(predictions):
    return binary_metrics([row["label"] for row in predictions],
                          [row["fake_probability"] for row in predictions])
