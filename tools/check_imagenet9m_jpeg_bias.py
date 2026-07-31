"""
Report the JPEG encoding (quality + chroma-subsampling) of the images in an
ImageNet-9M manifest, aggregated per target class (and per split).

By default it reads the *original* on-disk ImageNet files (``--source original``).
With ``--source biased`` it instead re-encodes each image with the (quality,
chroma-subsampling) of its assigned ``jpeg`` bias class (like
``my_datasets.utils.JPEGCompression``) and measures that.

Metadata reader:
  * `identify` (ImageMagick) if available -- uses the user's check_image().
  * otherwise a dependency-free PIL reader (quantization-table quality estimate
    + PIL.JpegImagePlugin.get_sampling for the subsampling).

Usage:
  python tools/check_imagenet9m_jpeg_bias.py \
      --csv data/imagenet9m/imagenet9m_single_jpeg_cls0-1_corr0.99_seed1.csv \
      --per-class 200            # sample size per (split, target); use --all for everything
"""

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from PIL.JpegImagePlugin import get_sampling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.cfg import CFG

# ImageNet-9 superclass names (kept local so this script has no heavy imports).
SUPERCLASS_NAMES = {
    0: "Dog",
    1: "Bird",
    2: "Vehicle",
    3: "Reptile",
    4: "Carnivore",
    5: "Insect",
    6: "Instrument",
    7: "Primate",
    8: "Fish",
}

_PIL_SUBSAMPLING = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0", -1: "grayscale"}
# ImageMagick sampling-factor strings -> canonical labels
_IDENTIFY_SUBSAMPLING = {
    "1x1,1x1,1x1": "4:4:4",
    "2x1,1x1,1x1": "4:2:2",
    "2x2,1x1,1x1": "4:2:0",
}


# --------------------------------------------------------------------------- #
# Produce the biased JPEG bytes (same encoding as JPEGCompression).
# --------------------------------------------------------------------------- #
def encode_biased_jpeg(path, quality, subsampling):
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(quality), subsampling=subsampling)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Metadata readers.
# --------------------------------------------------------------------------- #
def check_image_identify(image_path):
    """User-provided reader (ImageMagick `identify`). Returns (quality, subsampling)."""
    try:
        q = subprocess.run(
            ["identify", "-format", "%Q", image_path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        s = subprocess.run(
            ["identify", "-format", "%[jpeg:sampling-factor]", image_path],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        quality = float(q) if q else None
        sub = _IDENTIFY_SUBSAMPLING.get(s.replace(" ", ""), s)
        return quality, sub
    except subprocess.CalledProcessError:
        return None, "N/A"


def _build_quality_lut():
    """Map luminance quantization-table sum -> JPEG quality, using PIL's encoder."""
    rng = np.random.default_rng(0)
    sample = Image.fromarray(rng.integers(0, 256, (96, 96, 3)).astype("uint8"), "RGB")
    lut = {}
    for q in range(1, 101):
        buf = io.BytesIO()
        sample.save(buf, "JPEG", quality=q)
        buf.seek(0)
        im = Image.open(buf)
        lut[q] = sum(im.quantization[0])
    return lut


_QLUT = None


def check_image_pil(src):
    """Dependency-free reader. ``src`` is a file path or JPEG bytes.

    Returns (quality, subsampling); (None, "N/A") for non-JPEG / unreadable.
    """
    global _QLUT
    if _QLUT is None:
        _QLUT = _build_quality_lut()
    try:
        im = Image.open(src if isinstance(src, str) else io.BytesIO(src))
        if im.format != "JPEG" or not getattr(im, "quantization", None):
            return None, "N/A"
        sub = _PIL_SUBSAMPLING.get(get_sampling(im), f"other({get_sampling(im)})")
        qsum = sum(im.quantization[0])
        quality = min(_QLUT, key=lambda q: abs(_QLUT[q] - qsum))  # nearest match
        return float(quality), sub
    except Exception:
        return None, "N/A"


# --------------------------------------------------------------------------- #
def parse_classes_from_name(csv_path):
    """Recover the superclass ids from a '..._clsA-B-..._...' manifest filename."""
    m = re.search(r"_cls([\d-]+)_", os.path.basename(csv_path))
    return [int(x) for x in m.group(1).split("-")] if m else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--csv",
        default="data/imagenet9m/imagenet9m_single_jpeg_cls2-5_corr0.99_seed1.csv",
    )
    ap.add_argument(
        "--root", default=None, help="ImageNet root (default: cfg ROOT_IMAGENET)"
    )
    ap.add_argument(
        "--per-class",
        type=int,
        default=20000,
        help="samples per (split, target); ignored with --all",
    )
    ap.add_argument(
        "--all", action="store_true", help="use every image (slow with identify)"
    )
    ap.add_argument("--reader", choices=["auto", "identify", "pil"], default="auto")
    ap.add_argument(
        "--source",
        choices=["original", "biased"],
        default="original",
        help="'original' = on-disk ImageNet files; 'biased' = re-encode per assigned jpeg class",
    )
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    args = ap.parse_args()

    root = args.root or CFG.DATASET.IMAGENET9M.ROOT_IMAGENET
    jpeg_classes = [
        list(x) for x in CFG.DATASET.IMAGENET9M.JPEG_CLASSES
    ]  # index -> [quality, subsampling]
    classes = parse_classes_from_name(args.csv)  # superclass ids per target index

    reader = args.reader
    if reader == "auto":
        reader = "identify" if shutil.which("identify") else "pil"
    if reader == "identify" and not shutil.which("identify"):
        print("WARNING: `identify` not found on PATH; falling back to the PIL reader.")
        reader = "pil"

    df = pd.read_csv(args.csv)
    print(f"manifest : {args.csv}  ({len(df)} rows)")
    print(f"root     : {root}")
    print(f"reader   : {reader}")
    print(f"source   : {args.source}")
    if args.source == "biased":
        print(
            "jpeg bias classes (index -> quality/subsampling): "
            + ", ".join(f"{i}->{q}/{s}" for i, (q, s) in enumerate(jpeg_classes))
        )
    if classes:
        names = ", ".join(
            f"target {i} = superclass {c} ({SUPERCLASS_NAMES.get(c, '?')})"
            for i, c in enumerate(classes)
        )
        print(f"classes  : {names}")
    print()

    # accumulators keyed by (split, target) and (target,)
    qsum = defaultdict(float)
    qn = defaultdict(int)
    quals = defaultdict(Counter)
    subs = defaultdict(Counter)

    rng = np.random.default_rng(args.seed)
    for (split, target), grp in df.groupby(["split", "target"]):
        if not args.all and len(grp) > args.per_class:
            grp = grp.iloc[rng.choice(len(grp), args.per_class, replace=False)]
        for _, row in grp.iterrows():
            path = os.path.join(root, row["path"])

            # The data to measure: the original file, or biased re-encoded bytes.
            jbytes = None
            if args.source == "biased":
                quality_param, subsampling_param = jpeg_classes[int(row["jpeg"])]
                jbytes = encode_biased_jpeg(path, quality_param, subsampling_param)

            if reader == "identify":
                if jbytes is None:
                    quality, sub = check_image_identify(path)
                else:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                        tf.write(jbytes)
                        tmp = tf.name
                    try:
                        quality, sub = check_image_identify(tmp)
                    finally:
                        os.unlink(tmp)
            else:
                quality, sub = check_image_pil(path if jbytes is None else jbytes)

            for key in [(split, target), ("ALL", target)]:
                if quality is not None:
                    qsum[key] += quality
                    qn[key] += 1
                    quals[key][quality] += 1
                subs[key][sub] += 1

    # report
    def fmt_dist(counter, key_fmt=str):
        tot = sum(counter.values())
        return ", ".join(
            f"{key_fmt(k)}: {v / tot:.2%}" for k, v in counter.most_common()
        )

    def q_fmt(q):
        return str(int(q)) if float(q).is_integer() else f"{q:g}"

    for split in ["train", "val", "test", "ALL"]:
        keys = sorted(k for k in subs if k[0] == split)
        if not keys:
            continue
        header = "OVERALL (all splits)" if split == "ALL" else f"split = {split}"
        print(f"== {header} ==")
        for key in keys:
            target = key[1]
            name = SUPERCLASS_NAMES.get(classes[target], "?") if classes else "?"
            mean_q = qsum[key] / qn[key] if qn[key] else float("nan")
            n = sum(subs[key].values())
            print(f"  target {target} ({name}) | n={n} | mean quality={mean_q:.2f}")
            print(f"      quality     : {fmt_dist(quals[key], q_fmt)}")
            print(f"      subsampling : {fmt_dist(subs[key])}")
        print()


if __name__ == "__main__":
    main()
