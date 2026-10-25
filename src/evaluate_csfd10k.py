"""Evaluate ForenSort against the independent CSFD-10k MOS labels.

The evaluator reads grayscale images directly from the downloaded ZIP archive,
so the complete 4.9 GB uncompressed dataset does not need to be extracted.

Example:
    python src/evaluate_csfd10k.py \
        external_data/csfd10k/CSFD-images.zip \
        external_data/csfd10k/im_names.txt \
        external_data/csfd10k/mos.txt \
        outputs/checkpoints/baseline_best.pt

Use ``--all`` only when a complete run is intended. Without it, 100 evenly
spaced images are evaluated as a safe initial independent check.
"""

import argparse
import csv
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from inference import load_model, prepare_image, summarize_prediction


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


def read_labels(names_path: Path, mos_path: Path) -> list[tuple[str, float]]:
    names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
    mos = [float(line.strip()) for line in mos_path.read_text().splitlines() if line.strip()]
    if len(names) != len(mos):
        raise ValueError(f"Label count mismatch: {len(names)} names and {len(mos)} MOS values")
    if not names:
        raise ValueError("No CSFD-10k labels were found")
    return list(zip(names, mos))


def choose_indices(total: int, max_samples: int, evaluate_all: bool) -> list[int]:
    if evaluate_all:
        return list(range(total))
    if max_samples < 2:
        raise ValueError("--max-samples must be at least 2")
    count = min(total, max_samples)
    return np.linspace(0, total - 1, count, dtype=int).tolist()


def correlation(values: list[float], labels: list[float], function) -> float:
    result = function(values, labels)
    return float(result.statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("names", type=Path)
    parser.add_argument("mos", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--all", action="store_true", dest="evaluate_all")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/csfd10k_independent")
    )
    args = parser.parse_args()

    for path in (args.archive, args.names, args.mos, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    labels = read_labels(args.names, args.mos)
    indices = choose_indices(len(labels), args.max_samples, args.evaluate_all)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    records = []

    with zipfile.ZipFile(args.archive) as archive:
        names = set(archive.namelist())
        for batch_start in range(0, len(indices), args.batch_size):
            batch_indices = indices[batch_start:batch_start + args.batch_size]
            tensors = []
            boxes = []
            batch_labels = []
            for label_index in batch_indices:
                image_name, mos = labels[label_index]
                archive_name = f"CSFD-images/data/{image_name}"
                if archive_name not in names:
                    raise FileNotFoundError(
                        f"Image listed in labels is missing from archive: {image_name}"
                    )
                with archive.open(archive_name) as image_file:
                    image = np.asarray(Image.open(BytesIO(image_file.read())).convert("L"))
                tensor, _, box = prepare_image(image, args.size)
                tensors.append(tensor)
                boxes.append(box)
                batch_labels.append((image_name, mos))

            with torch.inference_mode():
                logits = model(torch.cat(tensors).to(device))
            probabilities = logits.softmax(1)
            predictions = probabilities.argmax(1).cpu().numpy().astype(np.uint8)
            confidences = probabilities.max(1).values.cpu().numpy().astype(np.float32)
            entropies = (
                -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)
                / np.log(6)
            ).cpu().numpy().astype(np.float32)

            for offset, (image_name, mos) in enumerate(batch_labels):
                prediction = predictions[offset]
                summary = summarize_prediction(
                    prediction,
                    confidences[offset],
                    entropies[offset],
                    content_mask(prediction.shape, boxes[offset], args.size),
                )
                records.append(
                    {
                        "image_name": image_name,
                        "mos": mos,
                        "forensort_image_score": summary["image_score"],
                        "forensort_average_quality": summary["average_quality"],
                        "forensort_usable_regions": summary["usable_regions"],
                    }
                )
            processed = batch_start + len(batch_indices)
            if processed == len(batch_indices) or processed % 25 == 0 or processed == len(indices):
                print(f"processed {processed}/{len(indices)}")

    scores = [row["forensort_image_score"] for row in records]
    mos_values = [row["mos"] for row in records]
    summary = {
        "dataset": "CSFD-10k",
        "samples": len(records),
        "selection": "all" if args.evaluate_all else "evenly_spaced",
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "spearman_forensort_score_vs_mos": correlation(scores, mos_values, spearmanr),
        "pearson_forensort_score_vs_mos": correlation(scores, mos_values, pearsonr),
        "mos_min": float(min(mos_values)),
        "mos_max": float(max(mos_values)),
        "forensort_score_min": float(min(scores)),
        "forensort_score_max": float(max(scores)),
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
