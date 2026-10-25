"""Evaluate a trained checkpoint on the untouched test split.

    python src/test.py configs/baseline.yaml outputs/checkpoints/baseline_best.pt

The test split is loaded only here, after training and model selection are
complete. Results are written beside the checkpoint unless --out is supplied.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from dataset import LatentQualityDataset, class_weights
from evaluate import evaluate
from model import QualityNet, masked_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, help="JSON output path")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = LatentQualityDataset(
        data_cfg["test_csv"] if "test_csv" in data_cfg else "data/manifests/test.csv",
        data_cfg["lffs_root"],
        data_cfg["size"],
        data_cfg["stride"],
        train=False,
    )
    test_dl = DataLoader(
        test_ds,
        train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = QualityNet(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    weights = class_weights(
        LatentQualityDataset(
            data_cfg["train_csv"],
            data_cfg["lffs_root"],
            data_cfg["size"],
            data_cfg["stride"],
            train=False,
        ).rows,
        train_cfg["class_weight_power"],
    ).to(device)
    metrics = evaluate(
        model,
        test_dl,
        device,
        lambda logits, target: masked_loss(
            logits,
            target,
            weights,
            ordinal_weight=train_cfg.get("ordinal_weight", 0.0),
        ),
    )
    metrics["checkpoint_epoch"] = checkpoint.get("epoch")
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["samples"] = len(test_ds)

    out = args.out or args.checkpoint.parent.parent / "metrics" / "baseline_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"test samples: {len(test_ds)}")
    print(
        f"test loss {metrics['loss']:.3f} | macro-F1 {metrics['macro_f1']:.3f} "
        f"| QWK {metrics['qwk']:.3f} | MAE {metrics['mae_levels']:.2f} "
        f"| accuracy {metrics['accuracy']:.3f}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
