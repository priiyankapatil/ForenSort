"""Compare ForenSort with OpenLQM on the held-out test split.

Example:
    python src/compare_openlqm.py \
        configs/baseline.yaml outputs/checkpoints/baseline_best.pt
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from ebts import parse_lffs
from inference import load_model, prepare_image, predict_with_confidence, summarize_prediction


OPENLQM_PATTERN = re.compile(
    r"(?P<path>.+?)\s+(?P<oq>-?\d+)\s+(?P<vid>-?\d+)\s+(?P<vcmp>-?\d+)\s*$"
)


def content_mask(shape, box, size):
    x0, y0, nw, nh = box
    stride_y = size / shape[0]
    stride_x = size / shape[1]
    ys = (np.arange(shape[0]) + 0.5) * stride_y
    xs = (np.arange(shape[1]) + 0.5) * stride_x
    return (
        (ys[:, None] >= y0)
        & (ys[:, None] < y0 + nh)
        & (xs[None, :] >= x0)
        & (xs[None, :] < x0 + nw)
    )


def run_openlqm(executable, image_path, ppi, timeout_seconds=30):
    try:
        result = subprocess.run(
            [str(executable), "-p", str(ppi), str(image_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"OpenLQM timed out after {timeout_seconds}s for {image_path}") from exc
    if result.returncode != 0:
        raise RuntimeError(
            f"OpenLQM failed for {image_path} (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    for line in result.stdout.splitlines():
        match = OPENLQM_PATTERN.match(line.strip())
        if match:
            return {key: int(match.group(key)) for key in ("oq", "vid", "vcmp")}
    raise RuntimeError(f"Could not parse OpenLQM output for {image_path}: {result.stdout!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--openlqm",
        type=Path,
        default=Path(r"C:\Program Files\OpenLQM\bin\openlqm.exe"),
    )
    parser.add_argument("--ppi", type=int, default=1000, choices=(500, 1000, 2000, 4000))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/openlqm_comparison"))
    args = parser.parse_args()

    if not args.openlqm.is_file():
        raise FileNotFoundError(f"OpenLQM executable not found: {args.openlqm}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = cfg["data"]
    rows = list(csv.DictReader(Path(data_cfg["test_csv"]).open(newline="")))
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("The test manifest contains no rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    size = int(data_cfg["size"])
    root = Path(data_cfg["lffs_root"])
    records = []

    for index, row in enumerate(rows, start=1):
        lffs_path = root / row["path"]
        latent = parse_lffs(lffs_path)
        if latent.image is None or latent.quality is None:
            raise ValueError(f"Missing image or expert map: {lffs_path}")

        tensor, _, box = prepare_image(latent.image, size)
        prediction, confidence, entropy = predict_with_confidence(model, tensor, device)
        summary = summarize_prediction(
            prediction,
            confidence,
            entropy,
            content_mask(prediction.shape, box, size),
        )
        with tempfile.TemporaryDirectory(prefix="forensort_openlqm_") as temp_dir:
            image_path = Path(temp_dir) / f"{row['image_id']}.png"
            Image.fromarray(latent.image).save(image_path)
            try:
                metrics = run_openlqm(args.openlqm, image_path, args.ppi)
                openlqm_error = ""
            except RuntimeError as exc:
                metrics = {"oq": None, "vid": None, "vcmp": None}
                openlqm_error = str(exc)
        records.append(
            {
                "image_id": row["image_id"],
                "path": row["path"],
                "expert_mean_quality": float(latent.quality.mean()),
                "forensort_average_quality": summary["average_quality"],
                "forensort_image_score": summary["image_score"],
                "forensort_usable_regions": summary["usable_regions"],
                "openlqm_oq": metrics["oq"],
                "openlqm_vid": metrics["vid"],
                "openlqm_vcmp": metrics["vcmp"],
                "openlqm_error": openlqm_error,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"processed {index}/{len(rows)}")

    valid = [
        row
        for row in records
        if row["openlqm_oq"] is not None and row["forensort_image_score"] >= 0
    ]
    if len(valid) < 2:
        raise ValueError("At least two valid OpenLQM results are required for correlation")

    def correlation(left, right):
        result = spearmanr(
            [row[left] for row in valid],
            [row[right] for row in valid],
        )
        return float(result.statistic)

    summary = {
        "samples": len(records),
        "valid_openlqm_samples": len(valid),
        "openlqm_failures": len(records) - len(valid),
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "openlqm": str(args.openlqm),
        "spearman": {
            "openlqm_oq_vs_expert_mean_quality": correlation(
                "openlqm_oq", "expert_mean_quality"
            ),
            "forensort_score_vs_expert_mean_quality": correlation(
                "forensort_image_score", "expert_mean_quality"
            ),
            "openlqm_oq_vs_forensort_score": correlation(
                "openlqm_oq", "forensort_image_score"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    json_path = args.out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
