"""Scan SD 302h, verify checksums, and write a subject-level train/val/test manifest.

    python src/build_manifest.py --variant original/unmasked

Outputs (data/manifests/, git-ignored):
    dataset_manifest.csv  one row per latent: ids, metadata, image/grid size, per-level cell counts, split, exclusion reason
    train.csv / val.csv / test.csv
Splitting is by participant (subject id), so no person appears in two splits.
"""
import argparse
import csv
import hashlib
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ebts import parse_lffs

LEVELS = range(6)
FIELDS = (["image_id", "path", "subject", "activity", "hand", "technique", "img_h", "img_w", "grid_h", "grid_w"]
          + [f"n_level{k}" for k in LEVELS] + ["checksum_ok", "split", "exclusion_reason"])


def subject_split(subject, seed, val_frac, test_frac):
    h = int(hashlib.sha256(f"{seed}:{subject}".encode()).hexdigest(), 16) % 10_000 / 10_000
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"


def scan_one(args):
    path, lffs_root, expected = args
    p = Path(path)
    rel = p.relative_to(lffs_root).as_posix()
    name = p.stem.split("_")  # SUBJECT_ACTIVITY_HAND_ENCOUNTER_TECHNIQUE_...
    row = {"image_id": p.stem, "path": rel, "subject": name[0], "activity": name[1], "hand": name[2],
           "technique": name[4], "exclusion_reason": ""}
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    row["checksum_ok"] = "" if expected is None else int(sha == expected)
    try:
        L = parse_lffs(p, decode_image=False)
    except Exception as e:  # keep going; record why
        row["exclusion_reason"] = f"parse_error:{type(e).__name__}"
        return row
    row["img_h"], row["img_w"] = L.size
    if L.quality is None:
        row["exclusion_reason"] = "no_quality_map"
        return row
    row["grid_h"], row["grid_w"] = L.quality.shape
    c = Counter(L.quality.ravel().tolist())
    for k in LEVELS:
        row[f"n_level{k}"] = c.get(k, 0)
    if set(c) - set(LEVELS):
        row["exclusion_reason"] = f"unexpected_levels:{sorted(set(c) - set(LEVELS))}"
    elif sum(c.get(k, 0) for k in range(1, 6)) == 0:
        row["exclusion_reason"] = "empty_map"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/ebts"))
    ap.add_argument("--variant", default="original/unmasked", help="e.g. original/unmasked, original/masked, enhanced/masked")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=Path("data/manifests"))
    a = ap.parse_args()

    lffs_root = a.data / "latent" / "lffs"
    files = sorted((lffs_root / a.variant).rglob("*.lffs"))
    print(f"{len(files)} files in {a.variant}")

    csum = a.data / f"checksum_latent_lffs_{a.variant.replace('/', '_')}.csv"
    expected = {}
    if csum.exists():
        with open(csum, newline="") as f:
            expected = {r["filename"]: r["sha256"] for r in csv.DictReader(f)}
    else:
        print("WARNING: no checksum file found, skipping verification")

    jobs = [(str(p), str(lffs_root), expected.get(p.relative_to(lffs_root).as_posix())) for p in files]
    with ProcessPoolExecutor() as ex:
        rows = list(ex.map(scan_one, jobs, chunksize=64))

    for r in rows:
        r["split"] = subject_split(r["subject"], a.seed, a.val_frac, a.test_frac)

    a.out.mkdir(parents=True, exist_ok=True)
    for name, sub in [("dataset_manifest", rows)] + [(s, [r for r in rows if r["split"] == s and not r["exclusion_reason"]]) for s in ("train", "val", "test")]:
        with open(a.out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
            w.writeheader(); w.writerows(sub)

    ok = [r for r in rows if not r["exclusion_reason"]]
    print(f"Checksums verified OK: {sum(r['checksum_ok'] == 1 for r in rows)} / bad: {sum(r['checksum_ok'] == 0 for r in rows)}")
    print("Exclusions:", dict(Counter(r["exclusion_reason"] for r in rows if r["exclusion_reason"])) or "none")
    print("Subjects:", len({r["subject"] for r in rows}))
    for s in ("train", "val", "test"):
        sub = [r for r in ok if r["split"] == s]
        print(f"  {s:5s} images={len(sub):5d} subjects={len({r['subject'] for r in sub})}")
    tot = [sum(int(r[f"n_level{k}"]) for r in ok) for k in LEVELS]
    print("Cell share by level (all usable):", {k: f"{100 * t / sum(tot):.1f}%" for k, t in zip(LEVELS, tot)})
    for s in ("train", "val", "test"):
        t = [sum(int(r[f"n_level{k}"]) for r in ok if r["split"] == s) for k in LEVELS]
        print(f"  {s:5s}", [f"{100 * x / sum(t):.1f}" for x in t])
    hs = sorted(int(r["img_h"]) for r in ok); ws = sorted(int(r["img_w"]) for r in ok)
    print(f"Image height min/median/max: {hs[0]}/{hs[len(hs)//2]}/{hs[-1]}  width: {ws[0]}/{ws[len(ws)//2]}/{ws[-1]}")


if __name__ == "__main__":
    main()
