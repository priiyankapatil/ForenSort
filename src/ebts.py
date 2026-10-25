"""Minimal parser for NIST SD 302h .lffs (EBTS / ANSI-NIST) records.

A record is a sequence of logical records. Each starts with "<type>.001:<byte length>"
and (for tagged-field records) holds "tag:value" fields separated by GS (0x1D);
sub-fields use US (0x1F), items RS (0x1E).

Fields we use:
  9.308   ridge quality map: rows of digits (RS between rows), one char per cell
  9.309   "<cell size>US<method>"  (cell size observed = 20; verify vs TN 2367)
  9.300   region of interest (width, height, x, y, polygon)
  13.999  embedded PNG latent image
"""
import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

GS, RS, US, FS = "\x1d", "\x1e", "\x1f", "\x1c"


@dataclass
class Latent:
    image: np.ndarray | None    # (H, W) uint8 grayscale; None if decode_image=False
    quality: np.ndarray | None   # (gh, gw) uint8 grid, or None if the record has no map
    fields: dict                 # raw type-9 fields (str -> str)
    path: Path
    size: tuple = (0, 0)         # (height, width) of the embedded image


def _logical_records(b: bytes) -> dict:
    recs, pos = {}, 0
    while pos < len(b):
        colon = b.index(b":", pos)
        gs = b.index(b"\x1d", colon)
        length = int(b[colon + 1:gs])
        recs[b[pos:colon].decode()] = b[pos:pos + length]
        pos += length
    return recs


def _png_from_type13(rec: bytes) -> np.ndarray:
    marker = rec.index(b"13.999:") + len(b"13.999:")
    return np.array(Image.open(io.BytesIO(rec[marker:])).convert("L"))


def parse_lffs(path, decode_image=True) -> Latent:
    path = Path(path)
    recs = _logical_records(path.read_bytes())
    fields = {}
    for fld in re.split(f"[{FS}{GS}]", recs["9.001"].decode("latin1")):
        tag, _, val = fld.partition(":")
        fields[tag] = val
    grid = None
    if "9.308" in fields:
        rows = fields["9.308"].split(RS)
        grid = np.array([[int(c) for c in r] for r in rows if r], dtype=np.uint8)
    r13 = recs["13.001"]
    hdr = dict(f.split(":", 1) for f in r13[:r13.index(b"13.999")].decode("latin1").split(GS) if ":" in f)
    size = (int(hdr["13.007"]), int(hdr["13.006"]))
    image = _png_from_type13(r13) if decode_image else None
    return Latent(image, grid, fields, path, size)
