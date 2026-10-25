"""Sanity check: can the model memorise a handful of training images? (it must, or something is broken)

    python src/overfit_test.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LatentQualityDataset, class_weights, IGNORE
from model import QualityNet, masked_loss

torch.manual_seed(0)
ds = LatentQualityDataset("data/manifests/train.csv", size=256, train=False)
idx = list(range(0, 8 * 700, 700))
X, Y = zip(*[ds[i] for i in idx])
X, Y = torch.stack(X), torch.stack(Y)
w = class_weights(ds.rows)

model = QualityNet(pretrained=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
print("input", tuple(X.shape), "-> target", tuple(Y.shape))
t0 = time.time()
for step in range(1, 61):
    model.train()
    logits = model(X)
    loss = masked_loss(logits, Y, w)
    opt.zero_grad(); loss.backward(); opt.step()
    if step == 1 or step % 10 == 0:
        valid = Y != IGNORE
        acc = (logits.argmax(1)[valid] == Y[valid]).float().mean().item()
        print(f"step {step:2d}  loss {loss.item():.3f}  cell accuracy {acc:.3f}  ({time.time() - t0:.0f}s)")
print("output", tuple(logits.shape))
