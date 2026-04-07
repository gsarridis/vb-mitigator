import re
import pandas as pd
import os
import random
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from my_datasets.utils import get_sampling_weights
from torch.utils.data.sampler import WeightedRandomSampler


def print_set_statistics(name, dataset):
    labels = [label for _, label, _ in dataset]
    origins = [origin for _, _, origin in dataset]
    origin_label_pairs = [(origin, label) for _, label, origin in dataset]

    print(f"\n{name} Set Statistics")
    print("=" * (len(name) + 15))
    print(f"Total samples: {len(dataset)}")
    print(f"Label counts: {Counter(labels)}")
    print(f"Origin counts: {Counter(origins)}")
    print(f"Origin-label breakdown: {Counter(origin_label_pairs)}")


NIH_DATA_BATCHES = [
    [1335, 6],
    [3923, 13],
    [6585, 6],
    [9232, 3],
    [11558, 7],
    [13774, 25],
    [16051, 9],
    [18387, 34],
    [20945, 49],
    [24717, 0],
    [28173, 2],
    [30805, 0],
]

CHEXPERT_DATA_BATCHES = [21513, 43017, 64540, 64740]


def get_nih_path(filename):
    img_id = int(filename.split("_")[0])
    img_subid = filename.split("_")[1]
    img_subid = img_subid.split(".")[0]
    img_subid = int(img_subid)
    for i, (max_id, max_subid) in enumerate(NIH_DATA_BATCHES):
        # if img_id == 13774:
        #     print(img_id, img_subid, i, max_id, max_subid)
        if img_id < max_id:
            return os.path.join(
                f"/mnt/cephfs/home/gsarridis/datasets/chestxrays/images_0{str(i+1).zfill(2)}/images",
                filename,
            )
        elif img_id == max_id and img_subid <= max_subid:
            return os.path.join(
                f"/mnt/cephfs/home/gsarridis/datasets/chestxrays/images_0{str(i+1).zfill(2)}/images",
                filename,
            )
        elif img_id == max_id and img_subid > max_subid:

            return os.path.join(
                f"/mnt/cephfs/home/gsarridis/datasets/chestxrays/images_0{str(i+2).zfill(2)}/images",
                filename,
            )
    raise ValueError(f"Image ID {img_id} not found in NIH batches.")


def get_chexpert_path(filename):
    # img_id = int(filename.split("_")[0])
    img_id = re.search(r"patient(\d+)", filename)
    if img_id:
        img_id = int(img_id.group(1))
        # print("Patient ID:", patient_id)
    else:
        raise ValueError(f"Could not extract the image id")
    for i, max_id in enumerate(CHEXPERT_DATA_BATCHES):
        if img_id <= max_id:
            # print(
            #     os.path.join(
            #         f"datasets/chestxrays/images_0{str(i+1).zfill(2)}/images", filename
            #     )
            # )
            full_path = os.path.join(
                "/mnt/cephfs/home/gsarridis/datasets/chexpert_download/chexpertchestxrays-u20210408/",
                filename,
            )
            if i <= 2:
                full_path = full_path.replace(
                    "CheXpert-v1.0/train",
                    f"CheXpert-v1.0 batch {str(i+2)} (train {str(i+1)})",
                )
            else:
                full_path = full_path.replace(
                    "CheXpert-v1.0",
                    "CheXpert-v1.0 batch 1 (validate & csv)",
                )
            return full_path
    raise ValueError(f"Image ID {img_id} not found in CheXpert batches.")


def load_nih_samples(disease, nih_root):
    meta = pd.read_csv(os.path.join(nih_root, "Data_Entry_2017.csv"))
    train_files = set(
        open(os.path.join(nih_root, "train_val_list.txt")).read().splitlines()
    )
    test_files = set(open(os.path.join(nih_root, "test_list.txt")).read().splitlines())

    def has_disease(labels, target):
        return target in labels.split("|")

    train_pos, train_neg, test_pos, test_neg = [], [], [], []

    for _, row in meta.iterrows():
        fname = row["Image Index"]
        labels = row["Finding Labels"]
        label = 1 if has_disease(labels, disease) else 0
        img_path = get_nih_path(fname)

        if fname in train_files:
            (train_pos if label == 1 else train_neg).append((img_path, label, 0))
        elif fname in test_files:
            (test_pos if label == 1 else test_neg).append((img_path, label, 0))

    return train_pos, train_neg, test_pos, test_neg


def load_chexpert_samples(disease, chexpert_train_path, chexpert_test_path):
    train_df = pd.read_csv(chexpert_train_path)
    test_df = pd.read_csv(chexpert_test_path)

    def process(df):
        pos, neg = [], []
        for _, row in df.iterrows():
            label = row.get(disease)
            if pd.isna(label):
                continue
            img_path = get_chexpert_path(row["Path"])
            if label == 1.0:
                pos.append((img_path, 1, 1))
            elif label == 0.0:
                neg.append((img_path, 0, 1))
        return pos, neg

    train_pos, train_neg = process(train_df)
    test_pos, test_neg = process(test_df)
    return train_pos, train_neg, test_pos, test_neg


def create_custom_split(
    disease="Pneumothorax",
    positive_bias_dataset="chexpert",
    bias_ratio=0.9,
    num_train_samples=None,
    nih_root="/mnt/cephfs/home/gsarridis/datasets/chestxrays",
    chexpert_train="/mnt/cephfs/home/gsarridis/datasets/chexpert_download/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/train.csv",
    chexpert_test="/mnt/cephfs/home/gsarridis/datasets/chexpert_download/chexpertchestxrays-u20210408/CheXpert-v1.0 batch 1 (validate & csv)/valid.csv",
):
    # Load all samples
    nih_train_pos, nih_train_neg, nih_test_pos, nih_test_neg = load_nih_samples(
        disease, nih_root
    )
    chex_train_pos, chex_train_neg, chex_test_pos, chex_test_neg = (
        load_chexpert_samples(disease, chexpert_train, chexpert_test)
    )

    # Shuffle
    random.shuffle(nih_train_pos)
    random.shuffle(nih_train_neg)
    random.shuffle(chex_train_pos)
    random.shuffle(chex_train_neg)

    # Select biased sources
    def split_biased(pos_main, pos_other, neg_main, neg_other):
        total = (
            num_train_samples
            if num_train_samples
            else min(len(pos_main) + len(pos_other), len(neg_main) + len(neg_other))
        )
        per_class = total // 2
        num_pos_main = int(bias_ratio * per_class)
        num_pos_other = per_class - num_pos_main
        num_neg_main = int((1 - bias_ratio) * per_class)
        num_neg_other = per_class - num_neg_main

        train = (
            pos_main[:num_pos_main]
            + pos_other[:num_pos_other]
            + neg_main[:num_neg_main]
            + neg_other[:num_neg_other]
        )
        random.shuffle(train)
        return train

    if positive_bias_dataset == "chexpert":
        train_set = split_biased(
            chex_train_pos, nih_train_pos, chex_train_neg, nih_train_neg
        )
    else:
        train_set = split_biased(
            nih_train_pos, chex_train_pos, nih_train_neg, chex_train_neg
        )

    # Test set: Balanced, unbiased
    # num_test = min(len(chex_test_pos + nih_test_pos), len(chex_test_neg + nih_test_neg))
    # half = num_test // 2
    # test_set = random.sample(chex_test_pos + nih_test_pos, half) + random.sample(
    #     chex_test_neg + nih_test_neg, half
    # )
    # random.shuffle(test_set)
    test_set = chex_test_pos + nih_test_pos + chex_test_neg + nih_test_neg
    return train_set, test_set


class CustomChestXrayDataset(Dataset):
    def __init__(self, samples, transform=None):
        """
        samples: list of (image_path, label, origin_dataset)
        """
        self.samples = samples
        self.transform = transform
        # Extract targets and biases as attributes
        self.targets = [label for _, label, _ in self.samples]
        self.biases = [origin for _, _, origin in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, origin = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            raise
        if transforms is not None:
            image = self.transform(image)
        return {
            "index": idx,
            "inputs": image,
            "targets": torch.tensor(label, dtype=torch.long),
            "bias": origin,
        }


def get_chexpert_nih_loader(
    batch_size=64, n_workers=4, transform=None, sampler=None, bias_ratio=0.9
) -> None:
    target_resolution = (256, 256)
    transform_test = transform
    if transform_test is None:
        transform_test = transforms.Compose(
            [
                transforms.Resize(
                    (
                        int(target_resolution[0]),
                        int(target_resolution[1]),
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    transform_train = transform
    if transform_train is None:
        transform_train = transforms.Compose(
            [
                transforms.Resize(
                    (
                        int(target_resolution[0]),
                        int(target_resolution[1]),
                    )
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    train_dataset, test_dataset = create_custom_split(bias_ratio=bias_ratio)
    train_dataset = CustomChestXrayDataset(train_dataset, transform=transform_train)
    test_dataset = CustomChestXrayDataset(test_dataset, transform_test)
    if sampler == "weighted":
        weights = get_sampling_weights(
            torch.tensor(train_dataset.targets),
            torch.tensor(train_dataset.biases),
        )
        sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    else:
        sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True if sampler is None else False,
        num_workers=n_workers,
        sampler=sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
    )
    return (
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
        test_loader,
        test_dataset,
    )


if __name__ == "__main__":
    train_set, test_set = create_custom_split()
    print_set_statistics("Train", train_set)
    print_set_statistics("Test", test_set)
