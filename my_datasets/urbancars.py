"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict
import os
import glob
import torch
import random


from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class UrbanCars(Dataset):

    obj_name_list = [
        "urban",
        "country",
    ]

    bg_name_list = [
        "urban",
        "country",
    ]

    co_occur_obj_name_list = [
        "urban",
        "country",
    ]

    def __init__(
        self,
        root: str,
        split: str,
        group_label="both",
        transform=None,
        max_samples_per_subgroup: int = None,
    ):
        if split == "train":
            bg_ratio = 0.95
            co_occur_obj_ratio = 0.95
        elif split in ["val", "test"]:
            bg_ratio = 0.5
            co_occur_obj_ratio = 0.5
        else:
            raise NotImplementedError
        self.bg_ratio = bg_ratio
        self.co_occur_obj_ratio = co_occur_obj_ratio

        assert os.path.exists(root)

        super().__init__()
        assert group_label in ["bg", "co_occur_obj", "both"]
        self.transform = transform

        ratio_combination_folder_name = (
            f"bg-{bg_ratio}_co_occur_obj-{co_occur_obj_ratio}"
        )
        img_root = os.path.join(root, ratio_combination_folder_name, split)

        self.img_fpath_list = []
        self.obj_bg_co_occur_obj_label_list = []

        for obj_id, obj_name in enumerate(self.obj_name_list):
            for bg_id, bg_name in enumerate(self.bg_name_list):
                for co_occur_obj_id, co_occur_obj_name in enumerate(
                    self.co_occur_obj_name_list
                ):
                    dir_name = (
                        f"obj-{obj_name}_bg-{bg_name}_co_occur_obj-{co_occur_obj_name}"
                    )
                    dir_path = os.path.join(img_root, dir_name)
                    # print(dir_path)
                    assert os.path.exists(dir_path)

                    img_fpath_list = glob.glob(os.path.join(dir_path, "*.jpg"))
                    self.img_fpath_list += img_fpath_list

                    self.obj_bg_co_occur_obj_label_list += [
                        (obj_id, bg_id, co_occur_obj_id)
                    ] * len(img_fpath_list)

        self.obj_bg_co_occur_obj_label_list = torch.tensor(
            self.obj_bg_co_occur_obj_label_list, dtype=torch.long
        )

        if max_samples_per_subgroup is not None:
            self._limit_samples_per_subgroup(max_samples_per_subgroup)

        self.obj_label = self.obj_bg_co_occur_obj_label_list[:, 0]
        self.bg_label = self.obj_bg_co_occur_obj_label_list[:, 1]
        self.co_occur_obj_label = self.obj_bg_co_occur_obj_label_list[:, 2]
        self.targets = self.obj_label
        num_shortcut_category = 4
        shortcut_label = self.bg_label * 2 + self.co_occur_obj_label

        self.domain_label = shortcut_label
        self.set_num_group_and_group_array(num_shortcut_category, shortcut_label)

    def _limit_samples_per_subgroup(self, max_samples: int):
        """Limit the number of samples per subgroup (obj x bg x co_occur_obj)."""

        # Group indices by subgroup
        subgroup_to_indices = defaultdict(list)
        for idx, (obj_id, bg_id, co_obj_id) in enumerate(
            self.obj_bg_co_occur_obj_label_list.tolist()
        ):
            subgroup_key = (obj_id, bg_id, co_obj_id)
            subgroup_to_indices[subgroup_key].append(idx)

        # Select up to max_samples from each subgroup
        selected_indices = []
        for subgroup_key, indices in subgroup_to_indices.items():
            if len(indices) > max_samples:
                # Randomly sample max_samples
                selected = random.sample(indices, max_samples)
            else:
                selected = indices
            selected_indices.extend(selected)

        # Sort to maintain some order (optional)
        selected_indices.sort()

        # Filter all lists
        self.img_fpath_list = [self.img_fpath_list[i] for i in selected_indices]
        self.obj_bg_co_occur_obj_label_list = self.obj_bg_co_occur_obj_label_list[
            selected_indices
        ]

        # Recompute derived labels
        self.obj_label = self.obj_bg_co_occur_obj_label_list[:, 0]
        self.bg_label = self.obj_bg_co_occur_obj_label_list[:, 1]
        self.co_occur_obj_label = self.obj_bg_co_occur_obj_label_list[:, 2]
        self.targets = self.obj_label

        shortcut_label = self.bg_label * 2 + self.co_occur_obj_label
        self.domain_label = shortcut_label
        self.set_num_group_and_group_array(4, shortcut_label)

        print(
            f"Limited to {max_samples} samples per subgroup: {len(self)} total samples"
        )

    def _get_subsample_group_indices(self):
        bg_ratio = self.bg_ratio
        co_occur_obj_ratio = self.co_occur_obj_ratio

        num_img_per_obj_class = len(self) // len(self.obj_name_list)

        min_bg_ratio = min(1 - bg_ratio, bg_ratio)
        min_co_occur_obj_ratio = min(1 - co_occur_obj_ratio, co_occur_obj_ratio)
        min_size = int(min_bg_ratio * min_co_occur_obj_ratio * num_img_per_obj_class)

        assert min_size > 1

        indices = []

        for idx_obj in range(len(self.obj_name_list)):
            obj_mask = self.obj_bg_co_occur_obj_label_list[:, 0] == idx_obj
            for idx_bg in range(len(self.bg_name_list)):
                bg_mask = self.obj_bg_co_occur_obj_label_list[:, 1] == idx_bg
                for idx_co_occur_obj in range(len(self.co_occur_obj_name_list)):
                    co_occur_obj_mask = (
                        self.obj_bg_co_occur_obj_label_list[:, 2] == idx_co_occur_obj
                    )
                    mask = obj_mask & bg_mask & co_occur_obj_mask
                    subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                    random.shuffle(subgroup_indices)
                    sampled_subgroup_indices = subgroup_indices[:min_size]
                    indices += sampled_subgroup_indices

        return indices

    def set_num_group_and_group_array(self, num_shortcut_category, shortcut_label):
        self.num_group = len(self.obj_name_list) * num_shortcut_category
        self.group_array = self.obj_label * num_shortcut_category + shortcut_label

    def set_domain_label(self, shortcut_label):
        self.domain_label = shortcut_label

    def __len__(self):
        return len(self.img_fpath_list)

    def __getitem__(self, index):
        img_fpath = self.img_fpath_list[index]
        label = self.obj_label[index]
        bg_label = self.bg_label[index]
        cooc_obj_label = self.co_occur_obj_label[index]

        img = Image.open(img_fpath)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # data_dict = {
        #     "image": img,
        #     "label": label,
        # }

        # if self.return_group_index:
        #     data_dict["group_index"] = self.group_array[index]

        # if self.return_domain_label:
        #     data_dict["domain_label"] = self.domain_label[index]

        # return data_dict
        return {
            "inputs": img,
            "targets": label,
            "background": bg_label,
            "object": cooc_obj_label,
            "index": index,
        }

    def get_labels(self):
        return self.obj_bg_co_occur_obj_label_list

    def get_sampling_weights(self):
        group_counts = (
            (torch.arange(self.num_group).unsqueeze(1) == self.group_array)
            .sum(1)
            .float()
        )
        group_weights = len(self) / group_counts
        weights = group_weights[self.group_array]
        return weights


def get_urbancars_loader(
    root, batch_size=128, num_workers=4, transform=None, image_size=224, split="train"
):

    if split == "train":
        if transform == None:
            transform = transforms.Compose(
                [
                    transforms.Resize(image_size + 32),
                    transforms.RandomRotation(45),
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        # Load the train and test datasets with the custom dataset class
        train_dataset = UrbanCars(
            root=root, split="train", transform=transform, max_samples_per_subgroup=10
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        return train_loader, train_dataset
    else:
        if transform == None:
            transform = transforms.Compose(
                [
                    transforms.Resize(image_size + 32),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        if split == "val":
            test_dataset = UrbanCars(
                root=root,
                split=split,
                transform=transform,
                max_samples_per_subgroup=1000,
            )
        else:
            test_dataset = UrbanCars(root=root, split=split, transform=transform)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return test_loader, test_dataset
