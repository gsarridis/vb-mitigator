"""
Stage 2 of the RAISE camera-bias benchmark: assign a *processing* bias and
materialize the final dataset.

Input is the camera-correlated manifest from ``tools/raise_stats.py --out`` (cols:
file, nef, tiff, target, camera, camera_idx, aligned, split). This script:

  1. Assigns a processing bias class per sample with a configurable correlation
     to the target (train biased at ``--proc-rho``; val/test balanced), 1:1 with
     the target class -- exactly like the imagenet9m single-bias assignment.
  2. Develops each source image (TIFF via PIL, or NEF via rawpy), resizes it, and
     RE-ENCODES it with the processing of its assigned class (e.g. JPEG vs PNG),
     baking the artifact into the saved file.
  3. Writes the materialized images to ``--out-dir/images/<split>/<target>/`` and a
     final manifest ``--out-dir/manifest.csv`` with both bias columns
     (``camera`` + ``processing``).

The result is a 2-class benchmark with TWO spurious cues: camera (fixed by
selection) and processing (assigned here) -- ready for training/eval where the
metrics group by (target, camera, processing).

Example:
  python tools/raise_build_benchmark.py \
      --manifest data/raise/raise_camera_landscape-buildings_rho0.9_seed1.csv \
      --raw-root /data/RAISE/TIFF --raw-ext .TIF \
      --out-dir  data/raise/bench_lb \
      --proc-classes "JPEG:quality=90;PNG" --proc-rho 0.95 --image-size 256
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# proc class index -> (format, params). Class i is the aligned processing for target i.
DEFAULT_PROCESSING = [("JPEG", {"quality": 90}), ("PNG", {})]
_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "JPEG2000": "jp2", "TIFF": "tif"}


# --------------------------------------------------------------------------- #
def parse_proc_classes(spec):
    """'JPEG:quality=90;PNG' -> [("JPEG", {"quality":90}), ("PNG", {})]."""
    classes = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        fmt, _, params = item.partition(":")
        kw = {}
        for kv in params.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                kw[k.strip()] = int(v) if v.strip().lstrip("-").isdigit() else v.strip()
        classes.append((fmt.strip().upper(), kw))
    return classes


# --------------------------------------------------------------------------- #
# Processing-bias assignment (configurable correlation to the target).
# --------------------------------------------------------------------------- #
def assign_processing(df, n_classes, rho, seed):
    """Per-sample processing class. Train biased at rho; val/test balanced; 1:1 with target."""
    rng = np.random.default_rng(seed)
    proc = np.full(len(df), -1, dtype=int)
    tgt = df["target"].to_numpy()
    spl = df["split"].to_numpy()
    for split in ["train", "val", "test"]:
        for t in range(n_classes):
            pos = np.where((spl == split) & (tgt == t))[0]
            n = len(pos)
            if n == 0:
                continue
            if split == "train":
                n_al = int(round(rho * n))
                labels = np.empty(n, dtype=int)
                labels[:n_al] = t  # aligned: processing class == target
                if n - n_al > 0:
                    others = [c for c in range(n_classes) if c != t]
                    labels[n_al:] = rng.choice(others, size=n - n_al)
                rng.shuffle(labels)
            else:
                labels = rng.integers(0, n_classes, size=n)  # balanced
            proc[pos] = labels
    return proc


# --------------------------------------------------------------------------- #
# Develop + process a single source image.
# --------------------------------------------------------------------------- #
def load_source(path, kind):
    if kind == "nef":
        import rawpy  # optional dependency
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(no_auto_bright=True, output_bps=8)
        return Image.fromarray(rgb, "RGB")
    return Image.open(path).convert("RGB")  # tiff / any PIL-readable


def resize_square(img, size):
    if size <= 0:
        return img
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def process_and_save(img, fmt, params, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, fmt, **params)


# --------------------------------------------------------------------------- #
def materialize(args):
    proc_classes = parse_proc_classes(args.proc_classes)
    df = pd.read_csv(args.manifest)
    n_targets = int(df["target"].nunique())
    assert len(proc_classes) >= n_targets, (
        f"need >= {n_targets} processing classes (1:1 with targets), got {len(proc_classes)}"
    )

    df["processing"] = assign_processing(df, n_targets, args.proc_rho, args.seed)
    df["processing_aligned"] = (df["processing"] == df["target"]).astype(int)

    records = []
    n = len(df) if args.limit is None else min(args.limit, len(df))
    for i, row in df.head(n).iterrows():
        src = os.path.join(args.raw_root, str(row["file"]) + args.raw_ext)
        if not os.path.isfile(src):
            print(f"  [skip] missing source: {src}")
            continue
        fmt, params = proc_classes[int(row["processing"])]
        rel = os.path.join("images", row["split"], str(row["target"]),
                            f"{row['file']}.{_EXT.get(fmt, fmt.lower())}")
        out_path = os.path.join(args.out_dir, rel)
        if not (args.resume and os.path.isfile(out_path)):
            img = resize_square(load_source(src, args.source), args.image_size)
            process_and_save(img, fmt, params, out_path)
        records.append({
            "image": rel,
            "file": row["file"],
            "target": int(row["target"]),
            "camera": row["camera"],
            "camera_idx": int(row["camera_idx"]),
            "processing": int(row["processing"]),
            "processing_class": f"{fmt}:{params}",
            "split": row["split"],
        })
        if (len(records) % 200) == 0:
            print(f"  processed {len(records)}/{n} ...")

    out_df = pd.DataFrame(records)
    man_path = os.path.join(args.out_dir, "manifest.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    out_df.to_csv(man_path, index=False)

    print(f"\nmaterialized {len(out_df)} images -> {args.out_dir}")
    print(f"final manifest -> {man_path}")
    print(f"processing classes: " + ", ".join(f"{i}->{f}:{p}" for i, (f, p) in enumerate(proc_classes)))
    if len(out_df):
        print("realized counts per (split, target, processing):")
        print(out_df.groupby(["split", "target", "processing"]).size().to_string())
        tr = out_df[out_df.split == "train"]
        al = (tr["processing"] == tr["target"]).mean()
        print(f"train processing aligned fraction (realized rho) = {al:.3f}")
    return out_df


# --------------------------------------------------------------------------- #
# Loader for the materialized benchmark.
# --------------------------------------------------------------------------- #
class RaiseProcessedDataset(Dataset):
    """Reads the materialized benchmark. Item: {inputs, targets, camera, processing, index}."""

    def __init__(self, out_dir, split, transform):
        man = pd.read_csv(os.path.join(out_dir, "manifest.csv"))
        man = man[man["split"] == split].reset_index(drop=True)
        self.root = out_dir
        self.transform = transform
        self.paths = [os.path.join(out_dir, p) for p in man["image"]]
        self.targets = man["target"].to_numpy()
        self.camera = man["camera_idx"].to_numpy()
        self.processing = man["processing"].to_numpy()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "inputs": img,
            "targets": int(self.targets[index]),
            "camera": int(self.camera[index]),
            "processing": int(self.processing[index]),
            "index": index,
        }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="camera-correlated manifest from raise_stats.py --out")
    ap.add_argument("--raw-root", required=True, help="local dir holding the source images (by file id)")
    ap.add_argument("--raw-ext", default=".TIF", help="source extension, e.g. .TIF or .NEF")
    ap.add_argument("--source", choices=["tiff", "nef"], default="tiff", help="how to develop the source")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--proc-classes", default="JPEG:quality=90;PNG",
                    help="';'-separated FORMAT[:k=v,...] specs; index i is aligned with target i")
    ap.add_argument("--proc-rho", type=float, default=0.95, help="train processing<->target correlation")
    ap.add_argument("--image-size", type=int, default=256, help="square center-crop size (0 to keep original)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="skip images already written")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N rows (debug)")
    args = ap.parse_args()
    materialize(args)


if __name__ == "__main__":
    main()
