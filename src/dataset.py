"""PyTorch dataset: latent image + expert ridge-quality grid -> (image, target map).

Pipeline per sample (train and eval share steps 1-3; augmentation is train-only):
  1. parse .lffs -> grayscale image, quality grid (levels 0..5)
  2. letterbox: scale longest side to `size`, pad to a square (white); padding is ignored by the loss
  3. stretch the grid over the image with nearest-neighbour so labels stay discrete
  4. (train) small random rotation/shift, applied identically to image and target; mild brightness/contrast
  5. downsample the target by `stride` (nearest) to the model's output resolution

Returns:
  image  float32 (1, size, size), ImageNet-gray normalised
  target int64   (size/stride, size/stride), values 0..5 or IGNORE (255) for padding / outside the warp
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from ebts import parse_lffs

IGNORE = 255
NUM_CLASSES = 6
GRAY_MEAN, GRAY_STD = 0.449, 0.226  # ImageNet mean/std averaged to one channel


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def class_weights(manifest_rows, power=0.5):
    """Inverse-frequency weights (softened by `power`) from level cell counts, normalised to mean 1."""
    counts = np.array([sum(int(r[f"n_level{k}"]) for r in manifest_rows) for k in range(NUM_CLASSES)], float)
    w = (counts.sum() / counts) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class LatentQualityDataset(Dataset):
    def __init__(self, manifest_csv, lffs_root="data/ebts/latent/lffs",
                 size=512, stride=8, train=False, seed=0):
        self.rows = read_manifest(manifest_csv)
        self.root = Path(lffs_root)  # manifest paths are relative to this, e.g. original/unmasked/<subj>/<file>
        self.size, self.stride, self.train = size, stride, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def _letterbox(self, img, grid):
        h, w = img.shape
        s = self.size / max(h, w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        tgt = cv2.resize(grid, (nw, nh), interpolation=cv2.INTER_NEAREST)  # grid stretched over whole image
        canvas = np.full((self.size, self.size), 255, np.uint8)
        tcanvas = np.full((self.size, self.size), IGNORE, np.uint8)
        y0, x0 = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = img
        tcanvas[y0:y0 + nh, x0:x0 + nw] = tgt
        return canvas, tcanvas

    def _augment(self, img, tgt):
        r = self.rng
        ang, tx, ty = r.uniform(-10, 10), r.uniform(-0.05, 0.05) * self.size, r.uniform(-0.05, 0.05) * self.size
        M = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), ang, 1.0)
        M[:, 2] += (tx, ty)
        img = cv2.warpAffine(img, M, (self.size, self.size), flags=cv2.INTER_LINEAR, borderValue=255)
        tgt = cv2.warpAffine(tgt, M, (self.size, self.size), flags=cv2.INTER_NEAREST, borderValue=IGNORE)
        a, b = r.uniform(0.85, 1.15), r.uniform(-15, 15)  # contrast, brightness
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return img, tgt

    def __getitem__(self, i):
        row = self.rows[i]
        L = parse_lffs(self.root / row["path"])
        img, tgt = self._letterbox(L.image, L.quality)
        if self.train:
            img, tgt = self._augment(img, tgt)
        off = self.stride // 2
        tgt = tgt[off::self.stride, off::self.stride]  # centre-sample each stride x stride block
        x = (img.astype(np.float32) / 255.0 - GRAY_MEAN) / GRAY_STD
        return torch.from_numpy(x)[None], torch.from_numpy(tgt.astype(np.int64))


if __name__ == "__main__":
    # sanity check: build a batch, print stats, save a figure of image + target
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    ds = LatentQualityDataset(f"data/manifests/{split}.csv", train=(split == "train"))
    print(split, "samples:", len(ds), "| class weights:", class_weights(ds.rows).numpy().round(2))
    fig, ax = plt.subplots(2, 4, figsize=(14, 7))
    for j in range(4):
        x, t = ds[j * 500]
        print(j, "image", tuple(x.shape), f"mean {x.mean():.2f}", "| target", tuple(t.shape),
              "levels:", {int(k): int(v) for k, v in zip(*np.unique(t.numpy(), return_counts=True))})
        ax[0, j].imshow(x[0], cmap="gray"); ax[0, j].axis("off")
        m = np.ma.masked_equal(t.numpy(), IGNORE)
        ax[1, j].imshow(m, cmap="viridis", vmin=0, vmax=5, interpolation="nearest"); ax[1, j].axis("off")
    fig.tight_layout(); fig.savefig("outputs/figures/loader_check.png", dpi=90)
    print("wrote outputs/figures/loader_check.png")
