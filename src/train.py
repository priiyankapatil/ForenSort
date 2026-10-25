"""Train the ForenSort quality model.

    python src/train.py configs/baseline.yaml
    python src/train.py configs/baseline.yaml --set train.epochs=1 train.limit_train=64 data.size=256

Uses train.csv for learning and val.csv for checkpoint selection / early stopping only.
The test split is never loaded here.
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LatentQualityDataset, class_weights
from evaluate import evaluate
from model import QualityNet, masked_loss


def set_overrides(cfg, items):
    for it in items or []:
        key, val = it.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for p in path:
            node = node[p]
        node[last] = yaml.safe_load(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--set", nargs="*", help="override config values, e.g. train.epochs=1")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    set_overrides(cfg, a.set)
    d, t = cfg["data"], cfg["train"]

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = LatentQualityDataset(d["train_csv"], d["lffs_root"], d["size"], d["stride"], train=True, seed=cfg["seed"])
    val_ds = LatentQualityDataset(d["val_csv"], d["lffs_root"], d["size"], d["stride"], train=False)
    weights = class_weights(train_ds.rows, t["class_weight_power"]).to(device)  # from TRAIN counts only
    if t["limit_train"]:
        train_ds = Subset(train_ds, range(t["limit_train"]))
    if t["limit_val"]:
        val_ds = Subset(val_ds, range(t["limit_val"]))
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, t["batch_size"], shuffle=True, num_workers=d["num_workers"], pin_memory=pin,
                          drop_last=True, persistent_workers=d["num_workers"] > 0)
    val_dl = DataLoader(val_ds, t["batch_size"], shuffle=False, num_workers=d["num_workers"], pin_memory=pin)
    print(f"train {len(train_ds)} | val {len(val_ds)} | class weights {weights.cpu().numpy().round(2)}")

    model = QualityNet(pretrained=True).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters()], lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    scaler = torch.amp.GradScaler(enabled=pin)
    loss_fn = lambda lg, y: masked_loss(lg, y, weights, ordinal_weight=t.get("ordinal_weight", 0.0))

    out = Path(cfg["out_dir"]); (out / "checkpoints").mkdir(parents=True, exist_ok=True); (out / "metrics").mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoints" / f"{cfg['name']}_best.pt"
    log_path = out / "metrics" / f"{cfg['name']}_train_log.csv"
    best, bad = -1.0, 0
    with open(log_path, "w", newline="") as lf:
        w = csv.writer(lf); w.writerow(["epoch", "train_loss", "val_loss", "val_macro_f1", "val_qwk", "val_mae_levels", "val_acc", "seconds"])
        for epoch in range(1, t["epochs"] + 1):
            t0 = time.time()
            model.freeze_encoder(epoch <= t["freeze_epochs"])
            model.train()
            run, nb = 0.0, 0
            for x, y in train_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device.type, enabled=pin):
                    loss = loss_fn(model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                run += loss.item(); nb += 1
            sched.step()
            m = evaluate(model, val_dl, device, loss_fn)
            row = [epoch, run / max(nb, 1), m["loss"], m["macro_f1"], m["qwk"], m["mae_levels"], m["accuracy"], time.time() - t0]
            w.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row]); lf.flush()
            print(f"epoch {epoch:2d} | train loss {row[1]:.3f} | val loss {row[2]:.3f} | macro-F1 {row[3]:.3f} | QWK {row[4]:.3f} "
                  f"| MAE {row[5]:.2f} | {row[7]:.0f}s")
            score = m[t["select_metric"]]
            if score > best:
                best, bad = score, 0
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": m}, ckpt)
                (out / "metrics" / f"{cfg['name']}_val_best.json").write_text(json.dumps(m, indent=2))
            else:
                bad += 1
                if bad >= t["patience"]:
                    print(f"early stop (no {t['select_metric']} gain for {bad} epochs)")
                    break
    print(f"best val {t['select_metric']}: {best:.3f} -> {ckpt}")


if __name__ == "__main__":
    main()
