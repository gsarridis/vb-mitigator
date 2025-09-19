import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy

from .utils import LoopPadding, VideoID, load_value_file
from .utils import (
    MultiScaleCornerCrop,
    TemporalRandomCrop,
    ClassLabel,
    get_sampling_weights,
)
from .utils import (
    Compose,
    Normalize,
    Scale,
    CenterCrop,
    CornerCrop,
    MultiScaleCornerCrop,
    MultiScaleRandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)
from torch.utils.data.sampler import WeightedRandomSampler


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


# def accimage_loader(path):
#     try:
#         import accimage

#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def get_default_image_loader():
    # from torchvision import get_image_backend

    # if get_image_backend() == "accimage":
    #     return accimage_loader
    # else:
    #     return pil_loader
    return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, "image_{:05d}.jpg".format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, "r") as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data["labels"]:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_class_biases(data, bias_type="coarse"):
    class_biases_map = {}
    index = 0
    if bias_type == "coarse":
        bias_key = "coarse_scene_labels"
    elif bias_type == "indoor_outdoor":
        bias_key = "indoor_outdoor"
    else:
        raise ValueError("Unrecognized bias type")
    for class_bias in data[bias_key]:
        class_biases_map[class_bias] = index
        index += 1
    return class_biases_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data["database"].items():
        this_subset = value["subset"]
        if this_subset == subset:
            label = value["annotations"]["label"]
            video_names.append("{}/{}".format(label, key))
            annotations.append(value["annotations"])

    return video_names, annotations


def coarse_to_indoor_outdoor(s):
    if s.startswith("indoor"):
        return "indoor"
    elif s.startswith("outdoor"):
        return "outdoor"
    else:
        return "outdoor"


def make_dataset(
    root_path,
    annotation_path,
    subset,
    n_samples_for_each_video,
    sample_duration,
    bias_type="coarse",
    bias_th=0.0,
):
    targets = []
    biases = []
    data = load_annotation_data(annotation_path)
    bias_per_label = data.get("bias_per_label", {})
    video_names, annotations = get_video_names_and_annotations(data, subset)
    # ✅ Keep only classes whose bias percentage >= bias_th
    allowed_classes = {cls for cls, bias in bias_per_label.items() if bias >= bias_th}

    # ✅ Filter video_names and annotations accordingly
    video_names, annotations = (
        zip(
            *[
                (vn, ann)
                for vn, ann in zip(video_names, annotations)
                if ann["label"] in allowed_classes
            ]
        )
        if video_names
        else ([], [])
    )

    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    bias_to_idx = get_class_biases(data, bias_type)
    idx_to_bias = {}
    for name, bias in bias_to_idx.items():
        idx_to_bias[bias] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print("dataset loading [{}/{}]".format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        # n_frames_file_path = os.path.join(video_path, "n_frames")
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}  # add more if needed

        n_frames = int(
            sum(
                1
                for f in os.listdir(video_path)
                if os.path.splitext(f)[1].lower() in valid_ext
            )
        )
        # n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            "video": video_path,
            "segment": [begin_t, end_t],
            "n_frames": n_frames,
            "video_id": video_names[i].split("/")[1],
        }
        if len(annotations) != 0:
            sample["label"] = class_to_idx[annotations[i]["label"]]
            if bias_type == "coarse":
                sample["bias"] = bias_to_idx[annotations[i]["coarse_scene_label"]]
            elif bias_type == "indoor_outdoor":
                sample["bias"] = bias_to_idx[
                    coarse_to_indoor_outdoor(annotations[i]["coarse_scene_label"])
                ]

            else:
                raise ValueError("unrecognized bias type")
        else:
            sample["label"] = -1
            sample["bias"] = -1

        if n_samples_for_each_video == 1:
            sample["frame_indices"] = list(range(1, n_frames + 1))
            dataset.append(sample)
            targets.append(sample["label"])
            biases.append(sample["bias"])
        else:
            if n_samples_for_each_video > 1:
                step = max(
                    1,
                    math.ceil(
                        (n_frames - 1 - sample_duration)
                        / (n_samples_for_each_video - 1)
                    ),
                )
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j["frame_indices"] = list(
                    range(j, min(n_frames + 1, j + sample_duration))
                )
                dataset.append(sample_j)
                targets.append(sample_j["label"])
                biases.append(sample_j["bias"])

    return dataset, idx_to_class, targets, biases


class UCF101(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root_path,
        annotation_path,
        subset,
        n_samples_for_each_video=1,
        spatial_transform=None,
        temporal_transform=None,
        target_transform=None,
        sample_duration=16,
        get_loader=get_default_video_loader,
        vis=False,
        bias_type="coarse",
        bias_th=0.0,
    ):
        self.data, self.class_names, self.targets, self.biases = make_dataset(
            root_path,
            annotation_path,
            subset,
            n_samples_for_each_video,
            sample_duration,
            bias_type=bias_type,
            bias_th=bias_th,
        )

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.vis = vis
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]["video"]

        frame_indices = self.data[index]["frame_indices"]
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        # target = self.data[index]
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # if self.vis:
        #     return clip, target, path, frame_indices
        # else:
        # return clip, target
        return {
            "index": index,
            "inputs": clip,
            "targets": self.targets[index],
            "bias": self.biases[index],
        }

    def __len__(self):
        return len(self.data)


def get_ucf101(
    video_path,
    annotation_path,
    batch_size=64,
    n_workers=4,
    transform=None,
    split="train",
    sampler=None,
    bias_type="coarse",
    bias_th=0.0,
) -> None:

    initial_scale = 1.0
    n_scales = 5
    scale_step = 0.84089641525
    scales = [initial_scale]
    for i in range(1, n_scales):
        scales.append(scales[-1] * scale_step)
    mean = [110.63666788 / 255, 103.16065604 / 255, 96.29023126 / 255]
    std = [38.7568578 / 255, 37.88248729 / 255, 40.02898126 / 255]
    sample_size = 112
    sample_duration = 16
    norm_value = 1
    crop_method = MultiScaleCornerCrop(scales, sample_size)
    norm_method = Normalize(mean, std)
    spatial_transform = Compose(
        [crop_method, RandomHorizontalFlip(), ToTensor(norm_value), norm_method]
    )

    temporal_transform = TemporalRandomCrop(sample_duration)
    target_transform = ClassLabel()

    if split == "train":
        train_dataset = UCF101(
            video_path,
            annotation_path,
            "training",
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            bias_type=bias_type,
            bias_th=bias_th,
        )

        if sampler == "weighted":
            weights = get_sampling_weights(train_dataset.targets, train_dataset.biases)
            sampler = WeightedRandomSampler(
                weights, len(train_dataset), replacement=True
            )
        else:
            sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True if sampler is None else False,
            num_workers=n_workers,
            sampler=sampler,
        )
        return train_loader, train_dataset

    elif split == "val":
        spatial_transform = Compose(
            [
                Scale(sample_size),
                CenterCrop(sample_size),
                ToTensor(norm_value),
                norm_method,
            ]
        )
        temporal_transform = LoopPadding(sample_duration)
        target_transform = ClassLabel()

        val_dataset = UCF101(
            video_path,
            annotation_path,
            "validation",
            3,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=sample_duration,
            vis=False,
            bias_type=bias_type,
            bias_th=bias_th,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True,
        )

        return val_loader, val_dataset
    elif split == "test":
        spatial_transform = Compose(
            [
                Scale(int(sample_size / 1.0)),
                CornerCrop(sample_size, "c"),
                ToTensor(norm_value),
                norm_method,
            ]
        )
        temporal_transform = LoopPadding(sample_duration)
        target_transform = VideoID()

        test_dataset = UCF101(
            video_path,
            annotation_path,
            "testing",
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=sample_duration,
            bias_type=bias_type,
            bias_th=bias_th,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
        )
        return test_loader, test_dataset
    else:
        raise ValueError(f"split {split} not recognized")
