"""Phase 2 dataset audit + manifest builder (stdlib only).

Usage:
    python src/audit_dataset.py data/raw/<subset>
    python src/audit_dataset.py data/raw/<subset> --group-regex "^(\\d+)_" --write

What it does:
  1. Inventories files by extension (so we learn what the dataset really contains).
  2. Pairs each image with annotation files that share its filename stem.
  3. Assigns a group id per image (default: the stem itself) and splits whole
     groups into train/val/test, so related impressions never straddle splits.
  4. Optionally writes data/manifests/dataset_manifest.csv + train/val/test.csv.

The split is a hash of the group id, so it is deterministic and does not change
if files are added later. Test labels are never inspected here.
"""
import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".wsq", ".jp2"}
FIELDS = ["image_id", "image_path", "label_paths", "group", "split", "exclusion_reason"]


def split_for(group: str, seed: int, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def build_manifest(root: Path, group_regex, seed, val_frac, test_frac):
    files = [p for p in root.rglob("*") if p.is_file()]
    by_stem = defaultdict(list)
    for p in files:
        by_stem[p.stem].append(p)

    rows = []
    for p in sorted(files):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        labels = [q for q in by_stem[p.stem] if q != p and q.suffix.lower() not in IMAGE_EXTS]
        group = p.stem
        if group_regex:
            m = group_regex.search(p.stem)
            if m:
                group = m.group(1)
        reason = "" if labels else "no_matching_label_file"
        rows.append({
            "image_id": p.stem,
            "image_path": str(p.relative_to(root)),
            "label_paths": ";".join(str(q.relative_to(root)) for q in labels),
            "group": group,
            "split": split_for(group, seed, val_frac, test_frac),
            "exclusion_reason": reason,
        })
    return files, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--group-regex", help="regex with ONE capture group giving the subject/source id from the filename stem")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--write", action="store_true", help="write manifest CSVs to data/manifests")
    ap.add_argument("--out", type=Path, default=Path("data/manifests"))
    a = ap.parse_args()

    if not a.root.is_dir():
        raise SystemExit(f"Not a directory: {a.root}")
    rx = re.compile(a.group_regex) if a.group_regex else None
    files, rows = build_manifest(a.root, rx, a.seed, a.val_frac, a.test_frac)

    print(f"Files found: {len(files)}")
    for ext, n in Counter(p.suffix.lower() or "(none)" for p in files).most_common():
        print(f"  {ext:10s} {n}")
    print(f"Images: {len(rows)}  |  without label file: {sum(1 for r in rows if r['exclusion_reason'])}")
    print("Split counts (images):", dict(Counter(r["split"] for r in rows)))
    print("Distinct groups:", len({r['group'] for r in rows}))
    print("Sample rows:")
    for r in rows[:3]:
        print("  ", r)
    if rows and len({r["group"] for r in rows}) == len(rows):
        print("NOTE: every image is its own group. If the dataset has subject/finger ids in "
              "filenames or metadata, pass --group-regex so splits are leakage-safe.")

    if a.write:
        a.out.mkdir(parents=True, exist_ok=True)
        for name, subset in [("dataset_manifest", rows)] + [(s, [r for r in rows if r["split"] == s]) for s in ("train", "val", "test")]:
            with open(a.out / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(subset)
        print(f"Wrote manifests to {a.out}")


if __name__ == "__main__":
    main()
