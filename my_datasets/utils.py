import random
import torch
import os
import zipfile
import requests
from tqdm import tqdm
import gdown
import tarfile
import math
import numbers
import collections
import numpy as np
from PIL import Image, ImageOps

accimage = None


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_confusion_matrix(num_classes, targets, biases):
    confusion_matrix_org = torch.zeros(num_classes, num_classes)
    confusion_matrix_org_by = torch.zeros(num_classes, num_classes)
    for t, p in zip(targets, biases):
        confusion_matrix_org[p.long(), t.long()] += 1
        confusion_matrix_org_by[t.long(), p.long()] += 1

    confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum(1).unsqueeze(1)
    confusion_matrix_by = confusion_matrix_org_by / confusion_matrix_org_by.sum(
        1
    ).unsqueeze(1)
    # confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum()
    return confusion_matrix_org, confusion_matrix, confusion_matrix_by


def get_unsup_confusion_matrix(num_classes, targets, biases, marginals):
    confusion_matrix_org = torch.zeros(num_classes, num_classes).float()
    confusion_matrix_cnt = torch.zeros(num_classes, num_classes).float()
    for t, p, m in zip(targets, biases, marginals):
        confusion_matrix_org[p.long(), t.long()] += m
        confusion_matrix_cnt[p.long(), t.long()] += 1

    zero_idx = confusion_matrix_org == 0
    confusion_matrix_cnt[confusion_matrix_cnt == 0] = 1
    confusion_matrix_org = confusion_matrix_org / confusion_matrix_cnt
    confusion_matrix_org[zero_idx] = 1
    confusion_matrix_org = 1 / confusion_matrix_org
    confusion_matrix_org[zero_idx] = 0

    confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum(1).unsqueeze(1)
    # confusion_matrix = confusion_matrix_org / confusion_matrix_org.sum()
    return confusion_matrix_org, confusion_matrix


# def download_celeba(root):
#     """
#     Downloads the CelebA dataset and extracts it to the directory `root/celeba`.

#     Args:
#         root (str): Root directory where the CelebA dataset will be extracted.

#     Raises:
#         Exception: If the download or extraction fails.
#     """
#     # Define URLs and paths
#     dataset_url = (
#         "https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset"
#     )
#     download_path = os.path.join(root, "celeba.zip")
#     extract_path = root

#     # Ensure the root directory exists
#     os.makedirs(root, exist_ok=True)

#     try:
#         # Download the dataset with a progress bar
#         print("Downloading CelebA dataset...")
#         response = requests.get(dataset_url, stream=True)
#         response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

#         # Get the total size from headers
#         total_size = int(response.headers.get("content-length", 0))
#         chunk_size = 1024
#         num_chunks = total_size // chunk_size + 1

#         # Write the dataset to a file with a progress bar
#         with open(download_path, "wb") as file, tqdm(
#             desc="Downloading",
#             unit="KB",
#             unit_divisor=1024,
#         ) as pbar:
#             for chunk in response.iter_content(chunk_size=chunk_size):
#                 if chunk:
#                     file.write(chunk)
#                     pbar.update(len(chunk))
#         print(f"Downloaded CelebA dataset to {download_path}.")

#         # Extract the dataset
#         print(f"Extracting CelebA dataset to {extract_path}...")
#         with zipfile.ZipFile(download_path, "r") as zip_ref:
#             zip_ref.extractall(extract_path)

#         extracted_dir = os.path.join(extract_path, "img_align_celeba")
#         final_path = os.path.join(root, "celeba")
#         if os.path.exists(extracted_dir):
#             os.rename(extracted_dir, final_path)
#         print(f"Extraction complete. Dataset available at {final_path}.")

#         # Optionally delete the zip file
#         os.remove(download_path)
#         print(f"Deleted the downloaded zip file: {download_path}.")

#         # move the extracted files: list_attr_celeba.csv, list_bbox_celeba.csv, list_eval_partition.csv, and list_landmarks_align_celeba.csv to final path
#         extracted_files = [
#             "list_attr_celeba.csv",
#             "list_bbox_celeba.csv",
#             "list_eval_partition.csv",
#             "list_landmarks_align_celeba.csv",
#         ]
#         for file_name in extracted_files:
#             extracted_file_path = os.path.join(extract_path, file_name)
#             final_file_path = os.path.join(
#                 root, "celeba", file_name.replace(".csv", ".txt")
#             )
#             if os.path.exists(extracted_file_path):
#                 os.makedirs(os.path.dirname(final_file_path), exist_ok=True)
#                 os.rename(extracted_file_path, final_file_path)
#         print(f"Moved and renamed extracted files to {os.path.join(root, 'celeba')}.")
#     except requests.exceptions.RequestException as e:
#         print(f"Failed to download the dataset: {e}")
#         raise
#     except zipfile.BadZipFile as e:
#         print(f"Failed to extract the dataset: {e}")
#         raise
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         raise


def download_celeba_zip(root):

    extract_path = root
    download_path = os.path.join(root, "celeba.zip")
    data_url = "https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    # Ensure the directory exists
    os.makedirs(root, exist_ok=True)
    try:
        # Download the data file from Google Drive
        print(download_path)
        if not os.path.isfile(download_path):
            gdown.download(data_url, download_path, quiet=False)
            print(f"Downloaded data to {download_path}.")

        # Extract the dataset
        print(f"Extracting CelebA dataset images to {extract_path}...")
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        extracted_dir = os.path.join(extract_path, "img_align_celeba")
        final_path = os.path.join(root, "celeba")
        if os.path.exists(extracted_dir):
            os.rename(extracted_dir, final_path)
        print(f"Extraction complete. Dataset available at {final_path}.")

        # Optionally delete the zip file
        os.remove(download_path)
        print(f"Deleted the downloaded zip file: {download_path}.")
    except zipfile.BadZipFile as e:
        print(f"Failed to extract the dataset: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def download_celeba_anno(root):

    extract_path = root
    files = {
        "list_eval_partition.txt": "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
        "list_landmarks_celeba.txt": "0B7EVK8r0v71pTzJIdlJWdHczRlU",
        "list_attr_celeba.txt": "0B7EVK8r0v71pblRyaVFSWGxPY0U",
        "list_bbox_celeba.txt": "0B7EVK8r0v71pbThiMVRxWXZ4dU0",
        "list_landmarks_align_celeba.txt": "0B7EVK8r0v71pd0FJY3Blby1HUTQ",
        "identity_CelebA.txt": "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
    }
    for file, uid in files.items():
        download_path = os.path.join(root, "celeba", file)
        data_url = f"https://drive.google.com/uc?id={uid}"
        # Ensure the directory exists
        os.makedirs(os.path.join(root, "celeba"), exist_ok=True)

        try:
            # Download the data file from Google Drive
            gdown.download(data_url, download_path, quiet=False)
            print(f"Downloaded data to {download_path}.")

        except zipfile.BadZipFile as e:
            print(f"Failed to extract the dataset: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise


def download_utkface(root):
    """
    Downloads the UTKFace dataset and extracts it to the directory `root/utk_face`.

    Args:
        root (str): Root directory where the UTKFace dataset will be extracted.

    Raises:
        Exception: If the download or extraction fails.
    """
    # Define URLs and paths
    dataset_url = "https://www.kaggle.com/api/v1/datasets/download/jangedoo/utkface-new"
    download_path = os.path.join(root, "utkface.zip")
    extract_path = root

    # Ensure the root directory exists
    os.makedirs(root, exist_ok=True)

    try:
        # Download the dataset with a progress bar
        print("Downloading UTKFace dataset...")
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

        # Get the total size from headers
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024
        num_chunks = total_size // chunk_size + 1

        # Write the dataset to a file with a progress bar
        with open(download_path, "wb") as file, tqdm(
            desc="Downloading",
            unit="KB",
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
        print(f"Downloaded UTKFace dataset to {download_path}.")

        # Extract the dataset
        print(f"Extracting UTKFace dataset to {extract_path}...")
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        # Optionally delete the zip file
        os.remove(download_path)
        print(f"Deleted the downloaded zip file: {download_path}.")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download the dataset: {e}")
        raise
    except zipfile.BadZipFile as e:
        print(f"Failed to extract the dataset: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def download_waterbirds(root):

    extract_path = root
    download_path = os.path.join(root, "waterbirds.zip")
    data_url = "https://drive.google.com/uc?id=1xPNYQskEXuPhuqT5Hj4hXPeJa9jh7liL"
    # Ensure the directory exists
    os.makedirs(root, exist_ok=True)

    try:
        # Download the data file from Google Drive
        gdown.download(data_url, download_path, quiet=False)
        print(f"Downloaded data to {download_path}.")

        # Extract the dataset
        print(f"Extracting Waterbirds dataset to {extract_path}...")
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        # Optionally delete the zip file
        os.remove(download_path)
        print(f"Deleted the downloaded zip file: {download_path}.")
    except zipfile.BadZipFile as e:
        print(f"Failed to extract the dataset: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def download_imagenet9(root):
    """
    Downloads the Imagenet9 background challenge dataset and extracts it to the directory `root/imagenet9`.

    Args:
        root (str): Root directory where the Imagenet9 dataset will be extracted.

    Raises:
        Exception: If the download or extraction fails.
    """
    # Define URLs and paths
    dataset_url = "https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz"
    download_path = os.path.join(root, "imagenet9.zip")
    extract_path = root

    # Ensure the root directory exists
    os.makedirs(root, exist_ok=True)

    try:
        # Download the dataset with a progress bar
        print("Downloading Imagenet9 dataset...")
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

        # Get the total size from headers
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024
        num_chunks = total_size // chunk_size + 1

        # Write the dataset to a file with a progress bar
        with open(download_path, "wb") as file, tqdm(
            desc="Downloading",
            unit="KB",
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
        print(f"Downloaded Imagenet9 dataset to {download_path}.")

        # Extract the dataset
        print(f"Extracting Imagenet9 dataset to {extract_path}...")
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print(f"Extraction complete!")

        # Optionally delete the zip file
        os.remove(download_path)
        print(f"Deleted the downloaded zip file: {download_path}.")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download the dataset: {e}")
        raise
    except zipfile.BadZipFile as e:
        print(f"Failed to extract the dataset: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


def get_subsample_group_indices(imgs, targets, bias_targets, bias_rate):
    """
    Returns subsampled group indices based on the bias attribute and bias rate.
    """
    # assert subsample_which_shortcut in ["bias", "both"], "Unsupported shortcut type"

    # Calculate the minimum number of samples per subgroup
    num_samples = len(imgs)
    num_classes = targets.max().item() + 1
    num_biases = bias_targets.max().item() + 1
    min_size = int(num_samples / num_classes * (1 - bias_rate))

    # Initialize the indices list
    indices = []

    # Loop through the target and bias groups
    for target_class in range(num_classes):
        target_mask = targets == target_class
        for bias_class in range(num_biases):
            bias_mask = bias_targets == bias_class
            group_mask = target_mask & bias_mask

            group_indices = torch.nonzero(group_mask).squeeze().tolist()
            random.shuffle(group_indices)

            # if subsample_which_shortcut == "bias" or subsample_which_shortcut == "both":
            sampled_indices = group_indices[:min_size]
            indices.extend(sampled_indices)

    return indices


def get_sampling_weights(targets, *bias_targets_list):
    """
    Calculates and returns sampling weights for each sample, generalized for multiple biases.

    Args:
        targets (Tensor): Target class labels.
        *bias_targets_list: Arbitrary number of bias attribute label Tensors.

    Returns:
        Tensor: Sampling weights for each sample.
    """
    if isinstance(targets, list):
        targets = torch.tensor(targets)

    num_classes = targets.max().item() + 1

    num_bias_categories = [
        bias_targets.max().item() + 1 for bias_targets in bias_targets_list
    ]

    # Calculate total number of groups
    num_groups = num_classes * torch.prod(torch.tensor(num_bias_categories))

    # Initialize group counts
    group_counts = torch.zeros(num_groups)

    # Count samples per group
    for idx in range(len(targets)):
        target_class = targets[idx].item()
        bias_indices = [bias_targets[idx].item() for bias_targets in bias_targets_list]

        # Compute unique group ID
        group_id = target_class
        multiplier = num_classes
        for bias_index, num_bias in zip(bias_indices, num_bias_categories):
            group_id += bias_index * multiplier
            multiplier *= num_bias

        group_counts[group_id] += 1

    # Calculate weights for each group
    group_weights = len(targets) / group_counts
    weights = torch.zeros(len(targets))

    # Assign weights to samples
    for idx in range(len(targets)):
        target_class = targets[idx].item()
        bias_indices = [bias_targets[idx].item() for bias_targets in bias_targets_list]

        # Compute unique group ID
        group_id = target_class
        multiplier = num_classes
        for bias_index, num_bias in zip(bias_indices, num_bias_categories):
            group_id += bias_index * multiplier
            multiplier *= num_bias

        weights[idx] = group_weights[group_id]
    # print(weights)
    return weights


def load_value_file(file_path):
    with open(file_path, "r") as input_file:
        value = float(input_file.read().rstrip("\n\r"))

    return value


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (
            isinstance(size, collections.Iterable) and len(size) == 2
        )
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass


class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        return img.crop((x1, y1, x1 + tw, y1 + th))

    def randomize_parameters(self):
        pass


class CornerCrop(object):

    def __init__(self, size, crop_position=None):
        self.size = size
        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.crop_position = crop_position
        self.crop_positions = ["c", "tl", "tr", "bl", "br"]

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == "c":
            th, tw = (self.size, self.size)
            x1 = int(round((image_width - tw) / 2.0))
            y1 = int(round((image_height - th) / 2.0))
            x2 = x1 + tw
            y2 = y1 + th
        elif self.crop_position == "tl":
            x1 = 0
            y1 = 0
            x2 = self.size
            y2 = self.size
        elif self.crop_position == "tr":
            x1 = image_width - self.size
            y1 = 0
            x2 = image_width
            y2 = self.size
        elif self.crop_position == "bl":
            x1 = 0
            y1 = image_height - self.size
            x2 = self.size
            y2 = image_height
        elif self.crop_position == "br":
            x1 = image_width - self.size
            y1 = image_height - self.size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[
                random.randint(0, len(self.crop_positions) - 1)
            ]


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.p = random.random()


class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        scales,
        size,
        interpolation=Image.BILINEAR,
        crop_positions=["c", "tl", "tr", "bl", "br"],
    ):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

        self.crop_positions = crop_positions

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        if self.crop_position == "c":
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_size // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half
        elif self.crop_position == "tl":
            x1 = 0
            y1 = 0
            x2 = crop_size
            y2 = crop_size
        elif self.crop_position == "tr":
            x1 = image_width - crop_size
            y1 = 0
            x2 = image_width
            y2 = crop_size
        elif self.crop_position == "bl":
            x1 = 0
            y1 = image_height - crop_size
            x2 = crop_size
            y2 = image_height
        elif self.crop_position == "br":
            x1 = image_width - crop_size
            y1 = image_height - crop_size
            x2 = image_width
            y2 = image_height

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self.crop_positions[
            random.randint(0, len(self.crop_positions) - 1)
        ]


class MultiScaleRandomCrop(object):

    def __init__(self, scales, size, interpolation=Image.BILINEAR):
        self.scales = scales
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        crop_size = int(min_length * self.scale)

        image_width = img.size[0]
        image_height = img.size[1]

        x1 = self.tl_x * (image_width - crop_size)
        y1 = self.tl_y * (image_height - crop_size)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        img = img.crop((x1, y1, x2, y2))

        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.tl_x = random.random()
        self.tl_y = random.random()


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[: self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class ComposeTarget(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst


class ClassLabel(object):

    def __call__(self, target):
        return target["label"]


class VideoID(object):

    def __call__(self, target):
        return target["video_id"]
