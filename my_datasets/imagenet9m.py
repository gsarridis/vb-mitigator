"""
ImageNet-9M: a configurable bias-benchmark variant of ImageNet-9.

Unlike :mod:`my_datasets.imagenet9` (which carries no bias annotations and is
evaluated on the Background Challenge), ImageNet-9M *synthesizes* controllable
shortcuts on top of a **class-balanced** subset of ImageNet-9 so we can build
benchmarks that probe single- and multi-attribute bias.

Two bias types are supported, injected as image transforms:
  * ``jpeg``   -- re-encode the image at a given (quality, chroma-subsampling).
  * ``resize`` -- resample the image by a scale factor (and back), leaving a
                  characteristic up/down-sampling artifact.

Scenarios:
  * ``single`` -- K classes, each 1:1 tied to one bias class of a single bias
                  type. Train is biased with ``CORRELATION``; val/test balanced.
  * ``multi``  -- exactly 2 classes with two bias types (jpeg + resize), each
                  with 2 bias classes. Train follows the UrbanCars-style joint
                  distribution derived from ``CORRELATION_JPEG`` /
                  ``CORRELATION_RESIZE`` (product of the two per-bias
                  correlations); val/test balanced.

Reproducibility: the random class-balancing subsample/split and the per-sample
bias assignment are computed once and cached as CSV manifests under
``MANIFEST_DIR``; subsequent runs reload them verbatim.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Reuse the canonical 1000 -> 9 ImageNet-9 superclass mapping (single source of
# truth) by enumerating its remapped samples, instead of duplicating the dict.
from my_datasets.imagenet9 import ImageNet9LDataset

# Bias transforms are the shared ones from utils.py.
from my_datasets.utils import JPEGCompression, Rescale

# 9 ImageNet-9 superclasses (same ids/names as the imagenet9 builder branch).
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
# Bias transforms: reuse JPEGCompression / Rescale from my_datasets/utils.py.
# A Rescale(scale) changes the image size; the standard transform below then
# normalizes back to ``image_size``, leaving the characteristic resize artifact.
# --------------------------------------------------------------------------- #
def _default_transform(split, image_size):
    """Standard ImageNet-9 transform recipe (bias transforms are applied before this)."""
    scale = (image_size + 32) / image_size
    target = (image_size, image_size)
    resize = transforms.Resize((int(target[0] * scale), int(target[1] * scale)))
    crop = (
        transforms.RandomCrop(target)
        if split == "train"
        else transforms.CenterCrop(target)
    )
    return transforms.Compose(
        [
            resize,
            crop,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# Stage 1: class-balanced base pool (shared by all scenarios, cached).
# --------------------------------------------------------------------------- #
def build_base_manifest(
    root_imagenet, manifest_dir, seed=0, split_ratios=(0.7, 0.1, 0.2)
):
    """Balance the 9 superclasses to the min class count and split 70/10/20.

    Returns a DataFrame with columns ``[path, superclass, split]`` where ``path``
    is relative to ``root_imagenet``. Result is cached to
    ``{manifest_dir}/imagenet9m_base_seed{seed}.csv`` and reloaded if present.
    """
    os.makedirs(manifest_dir, exist_ok=True)
    cache = os.path.join(manifest_dir, f"imagenet9m_base_seed{seed}.csv")
    if os.path.isfile(cache):
        return pd.read_csv(cache)

    # ImageNet9LDataset.samples == [(abs_path, superclass), ...], already remapped
    # to 0..8 with the excluded (-1) classes dropped. No images are loaded here.
    train_root = os.path.join(root_imagenet, "train")
    base = ImageNet9LDataset(train_root)

    by_cls = defaultdict(list)
    for path, superclass in base.samples:
        by_cls[superclass].append(os.path.relpath(path, root_imagenet))

    min_count = min(len(paths) for paths in by_cls.values())
    rng = np.random.default_rng(seed)
    r_train, r_val, _ = split_ratios

    rows = []
    for superclass in sorted(by_cls):
        paths = sorted(by_cls[superclass])  # deterministic order before shuffle
        order = rng.permutation(len(paths))[:min_count]  # balance + shuffle
        selected = [paths[i] for i in order]
        n = len(selected)
        n_train = int(round(r_train * n))
        n_val = int(round(r_val * n))
        split_labels = (
            ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
        )
        for path, split in zip(selected, split_labels):
            rows.append((path, superclass, split))

    df = pd.DataFrame(rows, columns=["path", "superclass", "split"])
    df.to_csv(cache, index=False)
    return df


# --------------------------------------------------------------------------- #
# Stage 2: per-sample bias assignment (per benchmark config, cached).
# --------------------------------------------------------------------------- #
def _assign_single(rng, n, target, num_classes, correlation, split):
    """Bias-class labels for ``n`` samples of one target in the single scenario."""
    if split == "train":
        n_aligned = int(round(correlation * n))
        labels = np.empty(n, dtype=int)
        labels[:n_aligned] = target  # aligned: bias class == target
        n_conflict = n - n_aligned
        if n_conflict > 0:
            others = [c for c in range(num_classes) if c != target]
            labels[n_aligned:] = rng.choice(others, size=n_conflict)  # uniform conflict
        rng.shuffle(labels)
        print(f"Train (class:{target}) - BA: {n_aligned}, BC: {n_conflict}")
        return labels
    # val/test: balanced -- bias class uniform over all classes, independent of target
    return rng.integers(0, num_classes, size=n)


def _assign_multi(rng, n, target, corr_jpeg, corr_resize, split):
    """(jpeg, resize) bias-class labels for ``n`` samples of one target (2-class)."""
    other = 1 - target
    if split == "train":
        # Joint groups over (jpeg_aligned?, resize_aligned?): UrbanCars-style product.
        props = [
            corr_jpeg * corr_resize,
            corr_jpeg * (1 - corr_resize),
            (1 - corr_jpeg) * corr_resize,
            (1 - corr_jpeg) * (1 - corr_resize),
        ]
        counts = [int(np.floor(p * n)) for p in props]
        counts[0] += n - sum(counts)  # rounding remainder -> both-aligned group
        jpeg_aligned = [1, 1, 0, 0]
        resize_aligned = [1, 0, 1, 0]
        jpeg = np.empty(n, dtype=int)
        resize = np.empty(n, dtype=int)
        pos = 0
        for g, cnt in enumerate(counts):
            jpeg[pos : pos + cnt] = target if jpeg_aligned[g] else other
            resize[pos : pos + cnt] = target if resize_aligned[g] else other
            pos += cnt
        perm = rng.permutation(n)
        return jpeg[perm], resize[perm]
    # val/test: balanced -- jpeg and resize each uniform over {0, 1}, independent
    return rng.integers(0, 2, size=n), rng.integers(0, 2, size=n)


def _benchmark_cache_name(
    scenario, classes, bias_type, correlation, corr_jpeg, corr_resize, seed
):
    cls = "-".join(str(c) for c in classes)
    if scenario == "single":
        return (
            f"imagenet9m_single_{bias_type}_cls{cls}_corr{correlation}_seed{seed}.csv"
        )
    return f"imagenet9m_multi_cls{cls}_cj{corr_jpeg}_cr{corr_resize}_seed{seed}.csv"


def build_benchmark_manifest(
    base_df,
    scenario,
    classes,
    bias_type,
    correlation,
    corr_jpeg,
    corr_resize,
    num_jpeg_classes,
    num_resize_classes,
    seed,
    manifest_dir,
):
    """Assign per-sample bias class(es) for a benchmark; cache & reload by config.

    Returns a DataFrame ``[path, split, target, <bias cols>]`` where the bias
    columns are ``jpeg`` and/or ``resize`` (the bias-class index per sample).
    """
    num_classes = len(classes)
    if scenario == "single":
        assert bias_type in ("jpeg", "resize"), f"unknown BIAS_TYPE: {bias_type}"
        avail = num_jpeg_classes if bias_type == "jpeg" else num_resize_classes
        assert avail >= num_classes, (
            f"single scenario needs >= {num_classes} '{bias_type}' bias classes "
            f"(1:1 with CLASSES), but only {avail} are defined."
        )
        bias_names = [bias_type]
    elif scenario == "multi":
        assert num_classes == 2, "multi scenario requires exactly 2 CLASSES."
        assert (
            num_jpeg_classes >= 2 and num_resize_classes >= 2
        ), "multi scenario needs >= 2 jpeg and >= 2 resize bias classes."
        bias_names = ["jpeg", "resize"]
    else:
        raise ValueError(f"unknown SCENARIO: {scenario}")

    os.makedirs(manifest_dir, exist_ok=True)
    cache = os.path.join(
        manifest_dir,
        _benchmark_cache_name(
            scenario, classes, bias_type, correlation, corr_jpeg, corr_resize, seed
        ),
    )
    if os.path.isfile(cache):
        return pd.read_csv(cache)

    rng = np.random.default_rng(seed + 1)  # offset so base & benchmark rngs differ

    cls2tgt = {c: i for i, c in enumerate(classes)}
    sub = base_df[base_df["superclass"].isin(classes)].copy()
    sub["target"] = sub["superclass"].map(cls2tgt)
    # Fixed, deterministic ordering so assignment is reproducible across runs.
    sub = sub.sort_values(["split", "target", "path"]).reset_index(drop=True)

    cols = {b: np.full(len(sub), -1, dtype=int) for b in bias_names}
    split_arr = sub["split"].to_numpy()
    target_arr = sub["target"].to_numpy()
    for split in ["train", "val", "test"]:
        for target in range(num_classes):
            pos = np.nonzero((split_arr == split) & (target_arr == target))[0]
            if len(pos) == 0:
                continue
            if scenario == "single":
                cols[bias_type][pos] = _assign_single(
                    rng, len(pos), target, num_classes, correlation, split
                )
            else:
                jpeg, resize = _assign_multi(
                    rng, len(pos), target, corr_jpeg, corr_resize, split
                )
                cols["jpeg"][pos] = jpeg
                cols["resize"][pos] = resize

    for b in bias_names:
        sub[b] = cols[b]
    df = sub[["path", "split", "target"] + bias_names]
    df.to_csv(cache, index=False)
    return df


# --------------------------------------------------------------------------- #
# Dataset + loader.
# --------------------------------------------------------------------------- #
class ImageNet9MDataset(Dataset):
    """Applies the assigned bias transform(s) per sample, then the std transform."""

    def __init__(
        self,
        df_split,
        root_imagenet,
        scenario,
        bias_type,
        jpeg_classes,
        resize_classes,
        transform,
    ):
        self.transform = transform
        self.bias_names = ["jpeg", "resize"] if scenario == "multi" else [bias_type]

        # One transform instance per bias class, from utils.py.
        self.jpeg_transforms = [
            JPEGCompression(quality=q, subsampling=s) for q, s in jpeg_classes
        ]
        self.resize_transforms = [
            Rescale(scale_factor=float(s)) for s in resize_classes
        ]

        self.paths = [os.path.join(root_imagenet, p) for p in df_split["path"].tolist()]
        self.targets = df_split["target"].to_numpy()
        self.bias = {b: df_split[b].to_numpy() for b in self.bias_names}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")

        # Order matters for multi: resample first, then compress.
        if "resize" in self.bias_names:
            img = self.resize_transforms[int(self.bias["resize"][index])](img)
        if "jpeg" in self.bias_names:
            img = self.jpeg_transforms[int(self.bias["jpeg"][index])](img)

        if self.transform is not None:
            img = self.transform(img)

        out = {"inputs": img, "targets": int(self.targets[index]), "index": index}
        for b in self.bias_names:
            out[b] = int(self.bias[b][index])
        return out


def get_imagenet9m_loader(
    cfg, split, transform=None, batch_size=None, workers=4, shuffle=None
):
    """Build the (loader, dataset) for one split of an ImageNet-9M benchmark."""
    c = cfg.DATASET.IMAGENET9M
    root = c.ROOT_IMAGENET
    classes = list(c.CLASSES)
    jpeg_classes = [list(x) for x in c.JPEG_CLASSES]
    resize_classes = [float(x) for x in c.RESIZE_CLASSES]

    base_df = build_base_manifest(
        root, c.MANIFEST_DIR, cfg.EXPERIMENT.SEED, tuple(c.SPLIT_RATIOS)
    )
    bench_df = build_benchmark_manifest(
        base_df,
        c.SCENARIO,
        classes,
        c.BIAS_TYPE,
        c.CORRELATION,
        c.CORRELATION_JPEG,
        c.CORRELATION_RESIZE,
        len(jpeg_classes),
        len(resize_classes),
        cfg.EXPERIMENT.SEED,
        c.MANIFEST_DIR,
    )
    df_split = bench_df[bench_df["split"] == split].reset_index(drop=True)

    if transform is None:
        transform = _default_transform(split, c.IMAGE_SIZE)

    dataset = ImageNet9MDataset(
        df_split, root, c.SCENARIO, c.BIAS_TYPE, jpeg_classes, resize_classes, transform
    )

    if batch_size is None:
        batch_size = cfg.SOLVER.BATCH_SIZE
    if shuffle is None:
        shuffle = split == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )
    return loader, dataset


def imagenet9m_bias_names(cfg):
    """Reported bias-attribute names for the configured scenario."""
    c = cfg.DATASET.IMAGENET9M
    return ["jpeg", "resize"] if c.SCENARIO == "multi" else [c.BIAS_TYPE]
