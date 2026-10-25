"""Build the two zips to upload to Google Drive for Colab training.

    python scripts/package_for_colab.py            # both zips
    python scripts/package_for_colab.py --code     # code only (re-upload after code changes; ~30 KB)

colab_upload/forensort_code.zip  src/, configs/, requirements.txt
colab_upload/forensort_data.zip  original/unmasked .lffs files + manifests (~2.3 GB, stored uncompressed: PNGs don't shrink)

Both unzip into one folder with the same layout as this project, so configs work unchanged.
The data zip contains restricted NIST data: upload only to your own private Drive.
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "colab_upload"


def add_tree(z, base, rel_root, patterns=("*",)):
    for pat in patterns:
        for p in sorted((ROOT / base).rglob(pat)):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(ROOT).as_posix())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", action="store_true", help="only build the code zip")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    with zipfile.ZipFile(OUT / "forensort_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        add_tree(z, "src", None, ("*.py",))
        add_tree(z, "configs", None, ("*.yaml",))
        z.write(ROOT / "requirements.txt", "requirements.txt")
    print("wrote", OUT / "forensort_code.zip")
    if a.code:
        return

    n = 0
    with zipfile.ZipFile(OUT / "forensort_data.zip", "w", zipfile.ZIP_STORED) as z:
        add_tree(z, "data/manifests", None, ("*.csv",))
        for p in sorted((ROOT / "data/ebts/latent/lffs/original/unmasked").rglob("*.lffs")):
            z.write(p, p.relative_to(ROOT).as_posix()); n += 1
    print(f"wrote {OUT / 'forensort_data.zip'}  ({n} .lffs files, {(OUT / 'forensort_data.zip').stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
