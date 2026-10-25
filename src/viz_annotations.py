"""Overlay the expert ridge-quality grid on latents so we can verify alignment by eye.

    python src/viz_annotations.py data/ebts/latent/lffs/original/unmasked --n 6 --out outputs/figures/annotations.png
"""
import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ebts import parse_lffs

CMAP = matplotlib.colors.ListedColormap(["none", "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("outputs/figures/annotations.png"))
    a = ap.parse_args()

    files = sorted(a.root.rglob("*.lffs"))
    random.Random(a.seed).shuffle(files)
    fig, axes = plt.subplots(2, a.n, figsize=(3 * a.n, 7))
    for i, f in enumerate(files[:a.n]):
        L = parse_lffs(f)
        axes[0, i].imshow(L.image, cmap="gray"); axes[0, i].set_title(f.name[:14], fontsize=7)
        axes[1, i].imshow(L.image, cmap="gray")
        if L.quality is not None:
            axes[1, i].imshow(L.quality, cmap=CMAP, vmin=0, vmax=5, alpha=0.5,
                              extent=(0, L.image.shape[1], L.image.shape[0], 0), interpolation="nearest")
        axes[1, i].set_title(f"grid {L.quality.shape}, max {L.quality.max()}", fontsize=7)
        for r in (0, 1):
            axes[r, i].axis("off")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
