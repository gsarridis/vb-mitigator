"""
SAE Combined Dataset for training SAEs on diverse visual concepts.

This dataset combines multiple source datasets for SAE training:
- CUB-200-2011: All bird images (200 species)
- Stanford Cars: All car images (196 types)
- Places365: Selected scene categories
- LVIS/COCO: Images containing selected object categories

The goal is to train an SAE that captures diverse visual features
including birds, cars, backgrounds/scenes, and common objects.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


# ============================================
# Individual Dataset Loaders
# ============================================


class CUB200Dataset(Dataset):
    """
    CUB-200-2011 Bird Dataset.

    Expected structure:
        root/
            images/
                001.Black_footed_Albatross/
                002.Laysan_Albatross/
                ...
            images.txt
            image_class_labels.txt
            train_test_split.txt
    """

    def __init__(
        self,
        root: str,
        split: str = "all",  # "train", "test", or "all"
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.split = split

        # Load image list
        images_file = self.root / "images.txt"
        if not images_file.exists():
            raise FileNotFoundError(f"CUB-200 images.txt not found at {images_file}")

        with open(images_file, "r") as f:
            self.images = {}
            for line in f:
                idx, path = line.strip().split()
                self.images[int(idx)] = path

        # Load class labels
        labels_file = self.root / "image_class_labels.txt"
        with open(labels_file, "r") as f:
            self.labels = {}
            for line in f:
                idx, label = line.strip().split()
                self.labels[int(idx)] = int(label) - 1  # 0-indexed

        # Load train/test split if needed
        if split != "all":
            split_file = self.root / "train_test_split.txt"
            with open(split_file, "r") as f:
                is_train = {}
                for line in f:
                    idx, flag = line.strip().split()
                    is_train[int(idx)] = int(flag) == 1

            if split == "train":
                self.indices = [i for i in self.images.keys() if is_train[i]]
            else:
                self.indices = [i for i in self.images.keys() if not is_train[i]]
        else:
            self.indices = list(self.images.keys())

        print(f"CUB-200: Loaded {len(self.indices)} images (split={split})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx]
        img_path = self.root / "images" / self.images[img_idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": self.labels[img_idx],
            "source": "cub200",
            "path": str(img_path),
        }


class StanfordCarsDataset(Dataset):
    """
    Stanford Cars Dataset.

    Expected structure (devkit version):
        root/
            cars_train/
            cars_test/
            devkit/
                cars_meta.mat
                cars_train_annos.mat
                cars_test_annos_withlabels.mat

    Or (Kaggle version):
        root/
            train/
            test/
            names.csv
            anno_train.csv
            anno_test.csv
    """

    def __init__(
        self,
        root: str,
        split: str = "all",
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.split = split

        self.samples = []

        # Try devkit format first
        devkit_path = self.root / "devkit"
        if devkit_path.exists() and loadmat is not None:
            self._load_devkit_format(split)
        else:
            # Try Kaggle/alternative format
            self._load_alternative_format(split)

        print(f"Stanford Cars: Loaded {len(self.samples)} images (split={split})")

    def _load_devkit_format(self, split):
        """Load from original devkit format."""
        if split in ["train", "all"]:
            annos = loadmat(self.root / "devkit" / "cars_train_annos.mat")
            annotations = annos["annotations"][0]
            for anno in annotations:
                img_name = anno["fname"][0]
                label = int(anno["class"][0][0]) - 1
                img_path = self.root / "cars_train" / img_name
                if img_path.exists():
                    self.samples.append((str(img_path), label))

        if split in ["test", "all"]:
            annos_file = self.root / "devkit" / "cars_test_annos_withlabels.mat"
            if annos_file.exists():
                annos = loadmat(annos_file)
                annotations = annos["annotations"][0]
                for anno in annotations:
                    img_name = anno["fname"][0]
                    label = int(anno["class"][0][0]) - 1
                    img_path = self.root / "cars_test" / img_name
                    if img_path.exists():
                        self.samples.append((str(img_path), label))

    def _load_alternative_format(self, split):
        """Load from alternative folder structure."""
        # Check for train/test folders with class subfolders
        for folder in ["train", "test", "cars_train", "cars_test"]:
            folder_path = self.root / folder
            if folder_path.exists():
                if split == "train" and "test" in folder:
                    continue
                if split == "test" and "train" in folder:
                    continue

                # Check if it has class subfolders
                subfolders = [d for d in folder_path.iterdir() if d.is_dir()]
                if subfolders:
                    for class_idx, class_folder in enumerate(sorted(subfolders)):
                        for img_path in class_folder.glob("*.jpg"):
                            self.samples.append((str(img_path), class_idx))
                        for img_path in class_folder.glob("*.png"):
                            self.samples.append((str(img_path), class_idx))
                else:
                    # Flat folder, label unknown
                    for img_path in folder_path.glob("*.jpg"):
                        self.samples.append((str(img_path), -1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "source": "stanford_cars",
            "path": img_path,
        }


class Places365FilteredDataset(Dataset):
    """
    Places365 Dataset filtered to specific categories.

    Expected structure:
        root/
            train/
                category_name/
                    *.jpg
            val/
                category_name/
                    *.jpg
            categories_places365.txt (optional)
    """

    # Default categories for bias-relevant backgrounds
    DEFAULT_CATEGORIES = [
        # Natural backgrounds (for waterbirds)
        "bamboo_forest",
        "forest/broadleaf",
        "forest-broadleaf",  # alternative naming
        "ocean",
        "lake/natural",
        "lake-natural",
        # Urban backgrounds (for cars)
        "alley",
        "crosswalk",
        "downtown",
        "gas_station",
        "garage/outdoor",
        "garage-outdoor",
        "driveway",
        # Road types
        "forest_road",
        "field_road",
        "desert_road",
        "highway",
    ]

    def __init__(
        self,
        root: str,
        categories: Optional[List[str]] = None,
        split: str = "all",
        transform: Optional[transforms.Compose] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.categories = categories or self.DEFAULT_CATEGORIES

        self.samples = []
        self.category_to_idx = {}

        # Normalize category names (handle / vs - variations)
        normalized_categories = set()
        for cat in self.categories:
            normalized_categories.add(cat)
            normalized_categories.add(cat.replace("/", "-"))
            normalized_categories.add(cat.replace("-", "/"))
            normalized_categories.add(cat.replace("/", "_"))
            normalized_categories.add(cat.replace("_", "/"))

        # Find matching folders
        folders_to_search = []
        if split in ["train", "all"]:
            train_dir = self.root / "train"
            if train_dir.exists():
                folders_to_search.append(train_dir)
            # Also check data_large for Places365
            data_large = self.root / "data_large"
            if data_large.exists():
                folders_to_search.append(data_large)

        if split in ["val", "test", "all"]:
            val_dir = self.root / "val"
            if val_dir.exists():
                folders_to_search.append(val_dir)

        # Search for matching categories
        found_categories = set()
        for base_dir in folders_to_search:
            for category_dir in base_dir.iterdir():
                if not category_dir.is_dir():
                    continue

                cat_name = category_dir.name
                # Check various naming conventions
                if (
                    cat_name in normalized_categories
                    or cat_name.lstrip("0123456789_") in normalized_categories
                ):

                    if cat_name not in self.category_to_idx:
                        self.category_to_idx[cat_name] = len(self.category_to_idx)

                    cat_idx = self.category_to_idx[cat_name]
                    found_categories.add(cat_name)

                    for img_path in category_dir.glob("*.jpg"):
                        self.samples.append((str(img_path), cat_idx, cat_name))
                    for img_path in category_dir.glob("*.png"):
                        self.samples.append((str(img_path), cat_idx, cat_name))

        print(f"Places365: Found {len(found_categories)} matching categories:")
        for cat in sorted(found_categories):
            count = sum(1 for s in self.samples if s[2] == cat)
            print(f"  - {cat}: {count} images")
        print(f"Places365: Total {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, category = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "category": category,
            "source": "places365",
            "path": img_path,
        }


class LVISFilteredDataset(Dataset):
    """
    LVIS Dataset filtered to specific object categories.
    Uses the official lvis package for proper annotation handling.

    Expected structure:
        root/
            lvis/
                lvis_v1_train.json
                lvis_v1_val.json
            coco/
                train2017/
                val2017/

    Requires: pip install lvis
    """

    # Default categories for bias-relevant objects
    DEFAULT_CATEGORIES = [
        "fire_hydrant",  # LVIS name for fireplug
        "fireplug",
        "stop_sign",
        "street_sign",
        "parking_meter",
        "traffic_light",
        "cow",
        "horse",
        "sheep",
    ]

    def __init__(
        self,
        lvis_root: str,
        coco_root: str,
        categories: Optional[List[str]] = None,
        split: str = "all",
        transform: Optional[transforms.Compose] = None,
        min_bbox_area: int = 0,  # Set > 0 to filter small objects
    ):
        self.lvis_root = Path(lvis_root)
        self.coco_root = Path(coco_root)
        self.transform = transform
        self.target_categories = set(
            cat.lower().replace(" ", "_")
            for cat in (categories or self.DEFAULT_CATEGORIES)
        )
        self.min_bbox_area = min_bbox_area

        # Setup image directories
        self.split_to_img_root = {
            "train": self.coco_root / "train2017",
            "val": self.coco_root / "val2017",
        }

        # Also check alternative paths
        if not self.split_to_img_root["train"].exists():
            alt_train = self.coco_root / "images" / "train2017"
            if alt_train.exists():
                self.split_to_img_root["train"] = alt_train
        if not self.split_to_img_root["val"].exists():
            alt_val = self.coco_root / "images" / "val2017"
            if alt_val.exists():
                self.split_to_img_root["val"] = alt_val

        self.samples = []  # (img_path, category_id, category_name)
        self.category_name_to_id = {}
        seen_images = set()

        # Try using lvis package first, fall back to manual JSON loading
        try:
            from lvis import LVIS

            self._load_with_lvis_package(split, LVIS, seen_images)
        except ImportError:
            print("LVIS package not found, using manual JSON loading")
            self._load_with_json(split, seen_images)

        print(f"LVIS: Loaded {len(self.samples)} images with target categories")
        self._print_category_stats()

    def _load_with_lvis_package(self, split: str, LVIS, seen_images: set):
        """Load using official lvis package."""
        splits_to_load = []
        if split in ["train", "all"]:
            train_json = self.lvis_root / "lvis_v1_train.json"
            if train_json.exists():
                splits_to_load.append(("train", str(train_json)))
        if split in ["val", "test", "all"]:
            val_json = self.lvis_root / "lvis_v1_val.json"
            if val_json.exists():
                splits_to_load.append(("val", str(val_json)))

        for split_name, json_path in splits_to_load:
            print(f"Loading LVIS {split_name} with lvis package...")
            lvis_api = LVIS(json_path)

            # Find matching category IDs
            matching_cat_ids = set()
            for cat in lvis_api.dataset["categories"]:
                cat_name = cat["name"].lower().replace(" ", "_")
                cat_id = cat["id"]

                for target in self.target_categories:
                    if cat_name == target or target in cat_name or cat_name in target:
                        matching_cat_ids.add(cat_id)
                        self.category_name_to_id[cat_name] = cat_id
                        print(f"  Matched: {cat_name} (id={cat_id})")
                        break

            # Filter annotations
            for ann in lvis_api.dataset["annotations"]:
                if ann["category_id"] not in matching_cat_ids:
                    continue

                # Optional: filter by bbox area
                if self.min_bbox_area > 0:
                    w, h = ann["bbox"][2:]
                    if w * h < self.min_bbox_area:
                        continue

                img_id = ann["image_id"]
                if img_id in seen_images:
                    continue

                # Find image path
                img_path = self._find_image_path(img_id)
                if img_path is not None:
                    cat_id = ann["category_id"]
                    # Reverse lookup for category name
                    cat_name = None
                    for name, cid in self.category_name_to_id.items():
                        if cid == cat_id:
                            cat_name = name
                            break
                    if cat_name is None:
                        cat_name = f"category_{cat_id}"

                    self.samples.append((str(img_path), cat_id, cat_name))
                    seen_images.add(img_id)

    def _load_with_json(self, split: str, seen_images: set):
        """Fallback: Load by parsing JSON directly."""
        splits_to_load = []
        if split in ["train", "all"]:
            train_json = self.lvis_root / "lvis_v1_train.json"
            if train_json.exists():
                splits_to_load.append(train_json)
        if split in ["val", "test", "all"]:
            val_json = self.lvis_root / "lvis_v1_val.json"
            if val_json.exists():
                splits_to_load.append(val_json)

        for json_path in splits_to_load:
            print(f"Loading LVIS from {json_path}...")
            with open(json_path, "r") as f:
                data = json.load(f)

            # Find matching category IDs
            matching_cat_ids = {}
            for cat in data["categories"]:
                cat_name = cat["name"].lower().replace(" ", "_")
                cat_id = cat["id"]

                for target in self.target_categories:
                    if cat_name == target or target in cat_name or cat_name in target:
                        matching_cat_ids[cat_id] = cat_name
                        self.category_name_to_id[cat_name] = cat_id
                        print(f"  Matched: {cat_name} (id={cat_id})")
                        break

            # Filter annotations
            for ann in data["annotations"]:
                cat_id = ann["category_id"]
                if cat_id not in matching_cat_ids:
                    continue

                if self.min_bbox_area > 0:
                    w, h = ann["bbox"][2:]
                    if w * h < self.min_bbox_area:
                        continue

                img_id = ann["image_id"]
                if img_id in seen_images:
                    continue

                img_path = self._find_image_path(img_id)
                if img_path is not None:
                    cat_name = matching_cat_ids[cat_id]
                    self.samples.append((str(img_path), cat_id, cat_name))
                    seen_images.add(img_id)

    def _find_image_path(self, img_id: int) -> Optional[Path]:
        """Find image path by trying train and val directories."""
        img_fname = f"{img_id:012d}.jpg"

        for split_name in ["train", "val"]:
            img_root = self.split_to_img_root.get(split_name)
            if img_root is None:
                continue
            img_path = img_root / img_fname
            if img_path.exists():
                return img_path

        return None

    def _print_category_stats(self):
        """Print category distribution."""
        category_counts = defaultdict(int)
        for _, _, cat_name in self.samples:
            category_counts[cat_name] += 1

        print(f"  Category distribution:")
        for name, count in sorted(category_counts.items()):
            print(f"    - {name}: {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, category_id, category_name = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": category_id,
            "category_name": category_name,
            "source": "lvis",
            "path": img_path,
        }


# ============================================
# Combined Dataset
# ============================================


class SAECombinedDataset(Dataset):
    """
    Combined dataset for SAE training.

    Merges CUB-200, Stanford Cars, Places365 (filtered), and LVIS (filtered)
    into a single dataset for training diverse SAEs.
    """

    def __init__(
        self,
        cub200_root: Optional[str] = None,
        stanford_cars_root: Optional[str] = None,
        places365_root: Optional[str] = None,
        lvis_root: Optional[str] = None,
        coco_root: Optional[str] = None,
        places_categories: Optional[List[str]] = None,
        lvis_categories: Optional[List[str]] = None,
        split: str = "all",
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            cub200_root: Path to CUB-200-2011 dataset
            stanford_cars_root: Path to Stanford Cars dataset
            places365_root: Path to Places365 dataset
            lvis_root: Path to LVIS annotations
            coco_root: Path to COCO images (for LVIS)
            places_categories: List of Places365 categories to include
            lvis_categories: List of LVIS categories to include
            split: "train", "val", "test", or "all"
            image_size: Size to resize images to
            transform: Optional custom transform
        """

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        else:
            self.transform = transform

        self.datasets = []
        self.dataset_names = []
        self.dataset_sizes = []

        # Load CUB-200
        if cub200_root and os.path.exists(cub200_root):
            print(f"\n{'='*50}")
            print("Loading CUB-200...")
            cub_dataset = CUB200Dataset(
                cub200_root, split=split, transform=self.transform
            )
            self.datasets.append(cub_dataset)
            self.dataset_names.append("cub200")
            self.dataset_sizes.append(len(cub_dataset))

        # Load Stanford Cars
        if stanford_cars_root and os.path.exists(stanford_cars_root):
            print(f"\n{'='*50}")
            print("Loading Stanford Cars...")
            cars_dataset = StanfordCarsDataset(
                stanford_cars_root, split=split, transform=self.transform
            )
            self.datasets.append(cars_dataset)
            self.dataset_names.append("stanford_cars")
            self.dataset_sizes.append(len(cars_dataset))

        # Load Places365 (filtered)
        if places365_root and os.path.exists(places365_root):
            print(f"\n{'='*50}")
            print("Loading Places365 (filtered)...")
            places_dataset = Places365FilteredDataset(
                places365_root,
                categories=places_categories,
                split=split,
                transform=self.transform,
            )
            if len(places_dataset) > 0:
                self.datasets.append(places_dataset)
                self.dataset_names.append("places365")
                self.dataset_sizes.append(len(places_dataset))

        # Load LVIS (filtered)
        if (
            lvis_root
            and coco_root
            and os.path.exists(lvis_root)
            and os.path.exists(coco_root)
        ):
            print(f"\n{'='*50}")
            print("Loading LVIS (filtered)...")
            lvis_dataset = LVISFilteredDataset(
                lvis_root,
                coco_root,
                categories=lvis_categories,
                split=split,
                transform=self.transform,
            )
            if len(lvis_dataset) > 0:
                self.datasets.append(lvis_dataset)
                self.dataset_names.append("lvis")
                self.dataset_sizes.append(len(lvis_dataset))

        # Compute cumulative sizes for indexing
        self.cumulative_sizes = []
        total = 0
        for size in self.dataset_sizes:
            total += size
            self.cumulative_sizes.append(total)

        print(f"\n{'='*50}")
        print("SAE Combined Dataset Summary:")
        print(f"{'='*50}")
        for name, size in zip(self.dataset_names, self.dataset_sizes):
            print(f"  {name}: {size} images")
        print(f"  TOTAL: {total} images")

    def __len__(self):
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumsum in enumerate(self.cumulative_sizes):
            if idx < cumsum:
                dataset_idx = i
                break

        # Compute local index within that dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Get item from the appropriate dataset
        item = self.datasets[dataset_idx][local_idx]

        return item

    def get_source_weights(self) -> torch.Tensor:
        """
        Get sampling weights to balance across source datasets.

        Returns weights that, when used with WeightedRandomSampler,
        will sample equally from each source dataset.
        """
        total = len(self)
        num_datasets = len(self.datasets)

        weights = []
        for size in self.dataset_sizes:
            # Weight inversely proportional to dataset size
            w = total / (num_datasets * size)
            weights.extend([w] * size)

        return torch.tensor(weights, dtype=torch.float)


def get_sae_combined_dataloader(
    cub200_root: Optional[str] = None,
    stanford_cars_root: Optional[str] = None,
    places365_root: Optional[str] = None,
    lvis_root: Optional[str] = None,
    coco_root: Optional[str] = None,
    places_categories: Optional[List[str]] = None,
    lvis_categories: Optional[List[str]] = None,
    split: str = "all",
    batch_size: int = 64,
    num_workers: int = 4,
    image_size: int = 224,
    balance_sources: bool = True,
    shuffle: bool = True,
) -> Tuple[DataLoader, SAECombinedDataset]:
    """
    Create dataloader for SAE combined dataset.

    Args:
        ... (same as SAECombinedDataset)
        batch_size: Batch size for dataloader
        num_workers: Number of worker processes
        balance_sources: If True, balance sampling across source datasets
        shuffle: Whether to shuffle (ignored if balance_sources=True)

    Returns:
        Tuple of (DataLoader, Dataset)
    """
    dataset = SAECombinedDataset(
        cub200_root=cub200_root,
        stanford_cars_root=stanford_cars_root,
        places365_root=places365_root,
        lvis_root=lvis_root,
        coco_root=coco_root,
        places_categories=places_categories,
        lvis_categories=lvis_categories,
        split=split,
        image_size=image_size,
    )

    if balance_sources and len(dataset) > 0:
        weights = dataset.get_source_weights()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(dataset), replacement=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return dataloader, dataset


# ============================================
# Default Categories
# ============================================

DEFAULT_PLACES_CATEGORIES = [
    # Natural (for waterbirds-like scenarios)
    "bamboo_forest",
    "forest/broadleaf",
    "ocean",
    "lake/natural",
    # Urban (for cars-like scenarios)
    "alley",
    "crosswalk",
    "downtown",
    "gas_station",
    "garage/outdoor",
    "driveway",
    # Roads
    "forest_road",
    "field_road",
    "desert_road",
]

DEFAULT_LVIS_CATEGORIES = [
    "fire_hydrant",  # fireplug
    "stop_sign",
    "street_sign",
    "parking_meter",
    "traffic_light",
    "cow",
    "horse",
    "sheep",
]


# ============================================
# Collate Function
# ============================================


def sae_combined_collate_fn(batch):
    """
    Custom collate function for SAECombinedDataset.

    Returns a dict compatible with the SAE mitigator's expected format:
    - inputs: batched image tensors
    - targets: class labels
    - source: source dataset indices (as "bias" attribute)
    - index: sample indices for visualization
    """
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    paths = [item["path"] for item in batch]
    sources = [item["source"] for item in batch]

    # Map sources to numeric indices for "bias" tracking
    source_to_idx = {"cub200": 0, "stanford_cars": 1, "places365": 2, "lvis": 3}
    source_indices = torch.tensor([source_to_idx.get(s, 0) for s in sources])

    return {
        "inputs": images,
        "targets": labels,
        "source": source_indices,  # As a "bias" attribute for analysis
        "sources": sources,  # String names
        "paths": paths,
        "index": torch.arange(len(batch)),
    }


# ============================================
# Integration with vb-mitigator
# ============================================


def get_sae_combined_dataloaders(cfg, transforms=None):
    """
    Create dataloaders for the SAE combined dataset.

    This function integrates with the vb-mitigator framework,
    allowing the SAE mitigator to use the combined dataset
    just like any other dataset.

    Config structure:
        DATASET:
          TYPE: "sae_combined"
          NUM_WORKERS: 8
          SAE_COMBINED:
            CUB200_ROOT: "/path/to/CUB_200_2011"
            STANFORD_CARS_ROOT: "/path/to/stanford_cars"
            PLACES365_ROOT: "/path/to/places365"
            LVIS_ROOT: "/path/to/lvis"
            COCO_IMAGES_ROOT: "/path/to/coco"
            PLACES365_CATEGORIES: [...]
            LVIS_CATEGORIES: [...]
            IMAGE_SIZE: 224
            SPLIT: "all"
            BALANCE_SOURCES: True
    """
    sae_cfg = cfg.DATASET.SAE_COMBINED

    # Get category lists (handle both list and tuple)
    places_cats = sae_cfg.get("PLACES365_CATEGORIES", DEFAULT_PLACES_CATEGORIES)
    lvis_cats = sae_cfg.get("LVIS_CATEGORIES", DEFAULT_LVIS_CATEGORIES)

    if hasattr(places_cats, "__iter__") and not isinstance(places_cats, str):
        places_cats = list(places_cats)
    if hasattr(lvis_cats, "__iter__") and not isinstance(lvis_cats, str):
        lvis_cats = list(lvis_cats)

    # Create dataset
    dataset = SAECombinedDataset(
        cub200_root=sae_cfg.get("CUB200_ROOT", None),
        stanford_cars_root=sae_cfg.get("STANFORD_CARS_ROOT", None),
        places365_root=sae_cfg.get("PLACES365_ROOT", None),
        lvis_root=sae_cfg.get("LVIS_ROOT", None),
        coco_root=sae_cfg.get("COCO_IMAGES_ROOT", None),
        places_categories=places_cats,
        lvis_categories=lvis_cats,
        split=sae_cfg.get("SPLIT", "all"),
        image_size=sae_cfg.get("IMAGE_SIZE", 224),
        transform=transforms,
    )

    # Create sampler if balancing sources
    sampler = None
    shuffle = True
    if sae_cfg.get("BALANCE_SOURCES", True):
        weights = dataset.get_source_weights()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(dataset), replacement=True
        )
        shuffle = False

    # Create train loader
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.SOLVER.BATCH_SIZE,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=True,
        collate_fn=sae_combined_collate_fn,
        drop_last=True,
    )

    return train_loader, dataset
