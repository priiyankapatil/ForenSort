"""Metrics for ordinal regional quality maps (plan section 7.2).

All metrics come from one confusion matrix over valid (non-padding) cells:
  macro-F1, per-class precision/recall/F1, quadratic weighted Cohen's kappa, mean absolute level error.
"""
import numpy as np
import torch

from dataset import IGNORE, NUM_CLASSES


@torch.no_grad()
def update_confusion(conf, logits, target):
    """conf: (C, C) long tensor, rows = expert level, cols = predicted level."""
    pred = logits.argmax(1)
    valid = target != IGNORE
    idx = target[valid] * NUM_CLASSES + pred[valid]
    conf += torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    return conf


def metrics_from_confusion(conf):
    c = conf.double().cpu().numpy()
    tp = np.diag(c)
    prec = tp / np.maximum(c.sum(0), 1)
    rec = tp / np.maximum(c.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    n = c.sum()
    levels = np.arange(NUM_CLASSES)
    w = (levels[:, None] - levels[None, :]) ** 2 / (NUM_CLASSES - 1) ** 2   # quadratic weights
    expected = np.outer(c.sum(1), c.sum(0)) / n
    kappa = 1 - (w * c).sum() / max((w * expected).sum(), 1e-12)
    mae = (np.abs(levels[:, None] - levels[None, :]) * c).sum() / n
    present = c.sum(1) > 0
    return {"macro_f1": float(f1[present].mean()), "qwk": float(kappa), "mae_levels": float(mae),
            "accuracy": float(tp.sum() / n), "f1_per_class": f1.round(3).tolist(),
            "recall_per_class": rec.round(3).tolist(), "precision_per_class": prec.round(3).tolist()}


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    tot_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if loss_fn is not None:
            tot_loss += loss_fn(logits, y).item() * x.size(0); n += x.size(0)
        update_confusion(conf, logits, y)
    m = metrics_from_confusion(conf)
    m["confusion"] = conf.cpu().tolist()
    if loss_fn is not None:
        m["loss"] = tot_loss / max(n, 1)
    return m
