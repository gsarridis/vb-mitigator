from my_datasets.celeba import get_celeba
from my_datasets.chexpert_nih import get_chexpert_nih_loader
from my_datasets.cifar100 import get_cifar100_loaders
from my_datasets.imagenet9 import get_background_challenge_data, get_imagenet9l
from my_datasets.speech_accent_archive import get_speech_accent_dataloaders
from my_datasets.ucf101 import get_ucf101
from my_datasets.urbancars import get_urbancars_loader
from my_datasets.urbansounds import get_urbansounds_dataloaders
from my_datasets.urbansounds58 import (
    get_urbansounds_dataloaders as get_urbansounds58_dataloaders,
)

from .biased_mnist import get_color_mnist
from .fb_biased_mnist import get_color_mnist as get_fb_color_mnist
from ram import get_transform
from .custom_transforms import get_transform as get_transform_padded
from .utk_face import get_utk_face
from .waterbirds import get_waterbirds
from .cifar10 import get_cifar10_loaders
from .stanford_dogs import get_stanford_dogs_loader
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from .imagenet import ImageNet
import torchvision


def get_dataset(cfg):
    dataset_name = cfg.DATASET.TYPE
    method_name = cfg.MITIGATOR.TYPE
    metric_name = cfg.METRIC

    if dataset_name == "biased_mnist":
        if method_name == "groupdro":
            train_loader, train_dataset = get_color_mnist(
                cfg.DATASET.BIASED_MNIST.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                data_label_correlation=cfg.DATASET.BIASED_MNIST.CORR,
                n_confusing_labels=9,
                split="train",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
                sampler="weighted",
            )
        else:
            train_loader, train_dataset = get_color_mnist(
                cfg.DATASET.BIASED_MNIST.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                data_label_correlation=cfg.DATASET.BIASED_MNIST.CORR,
                n_confusing_labels=9,
                split="train",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
            )

        val_loader, val_dataset = get_color_mnist(
            cfg.DATASET.BIASED_MNIST.ROOT,
            batch_size=256,
            data_label_correlation=0.1,
            n_confusing_labels=9,
            split="train_val",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )
        test_loader, test_dataset = get_color_mnist(
            cfg.DATASET.BIASED_MNIST.ROOT,
            batch_size=256,
            data_label_correlation=0.1,
            n_confusing_labels=9,
            split="valid",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )
        dataset = {}
        dataset["num_class"] = 10
        dataset["num_groups"] = 10 * 10
        dataset["biases"] = ["background"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        dataset["root"] = cfg.DATASET.BIASED_MNIST.ROOT
        dataset["target2name"] = {
            0: "number",
            1: "number",
            2: "number",
            3: "number",
            4: "number",
            5: "number",
            6: "number",
            7: "number",
            8: "number",
            9: "number",
        }
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
            or method_name == "mhmavias"
        ):
            tag_train_loader, _ = get_color_mnist(
                cfg.DATASET.BIASED_MNIST.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                data_label_correlation=cfg.DATASET.BIASED_MNIST.CORR,
                n_confusing_labels=9,
                split="train",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
                # transform=get_transform(
                #     image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                # ),
            )
            tag_test_loader, _ = get_color_mnist(
                cfg.DATASET.BIASED_MNIST.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                data_label_correlation=cfg.DATASET.BIASED_MNIST.CORR,
                n_confusing_labels=9,
                split="valid",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
                # transform=get_transform(
                #     image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                # ),
            )
            dataset["dataloaders"]["tag_test"] = tag_test_loader
            dataset["dataloaders"]["tag_train"] = tag_train_loader

    elif dataset_name == "fb_biased_mnist":
        if method_name == "groupdro":
            train_loader, train_dataset = get_fb_color_mnist(
                cfg.DATASET.FB_BIASED_MNIST.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                data_label_correlation1=cfg.DATASET.FB_BIASED_MNIST.CORR_BG,
                data_label_correlation2=cfg.DATASET.FB_BIASED_MNIST.CORR_FG,
                n_confusing_labels=9,
                split="train",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
                sampler="weighted",
            )
        else:
            train_loader, train_dataset = get_fb_color_mnist(
                cfg.DATASET.FB_BIASED_MNIST.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                data_label_correlation1=cfg.DATASET.FB_BIASED_MNIST.CORR_BG,
                data_label_correlation2=cfg.DATASET.FB_BIASED_MNIST.CORR_FG,
                n_confusing_labels=9,
                split="train",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
            )

        val_loader, val_dataset = get_fb_color_mnist(
            cfg.DATASET.FB_BIASED_MNIST.ROOT,
            batch_size=256,
            data_label_correlation1=0.1,
            data_label_correlation2=0.1,
            n_confusing_labels=9,
            split="train_val",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )
        test_loader, test_dataset = get_fb_color_mnist(
            cfg.DATASET.FB_BIASED_MNIST.ROOT,
            batch_size=256,
            data_label_correlation1=0.1,
            data_label_correlation2=0.1,
            n_confusing_labels=9,
            split="valid",
            seed=cfg.EXPERIMENT.SEED,
            aug=False,
        )

        dataset = {}
        dataset["num_class"] = 10
        dataset["num_groups"] = 10 * 10 * 10
        dataset["biases"] = ["background", "foreground"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        dataset["target2name"] = {
            0: "number",
            1: "number",
            2: "number",
            3: "number",
            4: "number",
            5: "number",
            6: "number",
            7: "number",
            8: "number",
            9: "number",
        }
        dataset["root"] = cfg.DATASET.FB_BIASED_MNIST.ROOT
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
        ):
            tag_train_loader, _ = get_fb_color_mnist(
                cfg.DATASET.BIASED_MNIST.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                data_label_correlation1=cfg.DATASET.FB_BIASED_MNIST.CORR_BG,
                data_label_correlation2=cfg.DATASET.FB_BIASED_MNIST.CORR_FG,
                n_confusing_labels=9,
                split="train",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            tag_test_loader, _ = get_fb_color_mnist(
                cfg.DATASET.BIASED_MNIST.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                data_label_correlation1=cfg.DATASET.FB_BIASED_MNIST.CORR_BG,
                data_label_correlation2=cfg.DATASET.FB_BIASED_MNIST.CORR_FG,
                n_confusing_labels=9,
                split="valid",
                seed=cfg.EXPERIMENT.SEED,
                aug=False,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            dataset["dataloaders"]["tag_train"] = tag_train_loader
            dataset["dataloaders"]["tag_test"] = tag_test_loader
    elif dataset_name == "utkface":
        if method_name == "groupdro":
            train_loader, train_dataset = get_utk_face(
                cfg.DATASET.UTKFACE.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                split="train",
                bias_attr=cfg.DATASET.UTKFACE.BIAS,
                image_size=cfg.DATASET.UTKFACE.IMAGE_SIZE,
                ratio=cfg.DATASET.UTKFACE.RATIO,
                sampler="weighted",
            )
        else:
            train_loader, train_dataset = get_utk_face(
                cfg.DATASET.UTKFACE.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                split="train",
                bias_attr=cfg.DATASET.UTKFACE.BIAS,
                image_size=cfg.DATASET.UTKFACE.IMAGE_SIZE,
                ratio=cfg.DATASET.UTKFACE.RATIO,
            )

        val_loader, val_dataset = get_utk_face(
            cfg.DATASET.UTKFACE.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            split="valid",
            bias_attr=cfg.DATASET.UTKFACE.BIAS,
            image_size=cfg.DATASET.UTKFACE.IMAGE_SIZE,
        )

        test_loader, test_dataset = get_utk_face(
            cfg.DATASET.UTKFACE.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            split="test",
            bias_attr=cfg.DATASET.UTKFACE.BIAS,
            image_size=cfg.DATASET.UTKFACE.IMAGE_SIZE,
        )

        dataset = {}
        dataset["num_class"] = 2
        dataset["num_groups"] = 2 * 2
        dataset["biases"] = [cfg.DATASET.UTKFACE.BIAS]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        dataset["target2name"] = {0: "male", 1: "female"}
        dataset["root"] = cfg.DATASET.UTKFACE.ROOT
        dataset["ba_groups"] = cfg.DATASET.UTKFACE.BIAS_ALIGNED
        # print(dataset["ba_groups"])
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
            or method_name == "mhmavias"
        ):
            tag_train_loader, _ = get_utk_face(
                cfg.DATASET.UTKFACE.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                split="train",
                bias_attr=cfg.DATASET.UTKFACE.BIAS,
                ratio=cfg.DATASET.UTKFACE.RATIO,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            tag_test_loader, _ = get_utk_face(
                cfg.DATASET.UTKFACE.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                split="test",
                bias_attr=cfg.DATASET.UTKFACE.BIAS,
                ratio=cfg.DATASET.UTKFACE.RATIO,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            dataset["dataloaders"]["tag_train"] = tag_train_loader
            dataset["dataloaders"]["tag_test"] = tag_train_loader
    elif dataset_name == "waterbirds":
        if method_name == "groupdro":
            train_loader, train_dataset = get_waterbirds(
                cfg.DATASET.WATERBIRDS.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                n_workers=cfg.DATASET.NUM_WORKERS,
                split="train",
                sampler="weighted",
            )
        else:
            train_loader, train_dataset = get_waterbirds(
                cfg.DATASET.WATERBIRDS.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                n_workers=cfg.DATASET.NUM_WORKERS,
                split="train",
            )

        val_loader, val_dataset = get_waterbirds(
            cfg.DATASET.WATERBIRDS.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            n_workers=cfg.DATASET.NUM_WORKERS,
            split="val",
        )
        test_loader, test_dataset = get_waterbirds(
            cfg.DATASET.WATERBIRDS.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            n_workers=cfg.DATASET.NUM_WORKERS,
            split="test",
        )

        dataset = {}
        dataset["num_class"] = 2
        dataset["biases"] = ["background"]
        dataset["num_groups"] = 2 * 2
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        dataset["root"] = cfg.DATASET.WATERBIRDS.ROOT
        dataset["target2name"] = {
            0: "landbird",
            1: "waterbird",
        }
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
        ):
            tag_train_loader, _ = get_waterbirds(
                cfg.DATASET.WATERBIRDS.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                n_workers=cfg.DATASET.NUM_WORKERS,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
                split="train",
            )
            tag_test_loader, _ = get_waterbirds(
                cfg.DATASET.WATERBIRDS.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                n_workers=cfg.DATASET.NUM_WORKERS,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
                split="test",
            )
            dataset["dataloaders"]["tag_train"] = tag_train_loader
            dataset["dataloaders"]["tag_test"] = tag_test_loader
    elif dataset_name == "celeba":
        if method_name == "groupdro":
            train_loader, train_dataset = get_celeba(
                cfg.DATASET.CELEBA.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                split="train",
                target_attr=cfg.DATASET.CELEBA.TARGET,
                img_size=cfg.DATASET.CELEBA.IMAGE_SIZE,
                ratio=cfg.DATASET.CELEBA.RATIO,
                sampler="weighted",
            )
        else:
            train_loader, train_dataset = get_celeba(
                cfg.DATASET.CELEBA.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                split="train",
                target_attr=cfg.DATASET.CELEBA.TARGET,
                img_size=cfg.DATASET.CELEBA.IMAGE_SIZE,
                ratio=cfg.DATASET.CELEBA.RATIO,
            )

        val_loader, val_dataset = get_celeba(
            cfg.DATASET.CELEBA.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            split="valid",
            target_attr=cfg.DATASET.CELEBA.TARGET,
            img_size=cfg.DATASET.CELEBA.IMAGE_SIZE,
        )

        test_loader, test_dataset = get_celeba(
            cfg.DATASET.CELEBA.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            split="test",
            target_attr=cfg.DATASET.CELEBA.TARGET,
            img_size=cfg.DATASET.CELEBA.IMAGE_SIZE,
        )

        dataset = {}
        dataset["num_class"] = 2
        dataset["num_groups"] = 2 * 2
        dataset["biases"] = [cfg.DATASET.CELEBA.BIAS]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        if cfg.DATASET.CELEBA.TARGET == "blonde":
            dataset["target2name"] = {0: "non_blonde", 1: "blonde"}
        elif cfg.DATASET.CELEBA.TARGET == "makeup":
            dataset["target2name"] = {0: "no_makeup", 1: "makeup"}
        else:
            raise ValueError("Target attribute should be either blonde or makeup.")
        dataset["root"] = cfg.DATASET.CELEBA.ROOT
        dataset["ba_groups"] = cfg.DATASET.CELEBA.BIAS_ALIGNED
        # print(dataset["ba_groups"])
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
            or method_name == "mhmavias"
        ):
            tag_train_loader, _ = get_celeba(
                cfg.DATASET.CELEBA.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                split="train",
                target_attr=cfg.DATASET.CELEBA.TARGET,
                ratio=cfg.DATASET.CELEBA.RATIO,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            tag_test_loader, _ = get_celeba(
                cfg.DATASET.CELEBA.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                split="test",
                target_attr=cfg.DATASET.CELEBA.TARGET,
                ratio=cfg.DATASET.CELEBA.RATIO,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            dataset["dataloaders"]["tag_train"] = tag_train_loader
            dataset["dataloaders"]["tag_test"] = tag_test_loader
    elif dataset_name == "imagenet9":
        if method_name == "groupdro":
            raise ValueError(
                "GroupDro requires bias attribute annotations! The imagenet dataset does not offer such information. Please select another method, or modify imagenet class so that it incorporates your own bias annotations."
            )
        else:
            train_loader, train_dataset = get_imagenet9l(
                root=cfg.DATASET.IMAGENET9.ROOT_IMAGENET,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                image_size=cfg.DATASET.IMAGENET9.IMAGE_SIZE,
            )

        val_loader = get_background_challenge_data(
            root=cfg.DATASET.IMAGENET9.ROOT_IMAGENET_BG,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.IMAGENET9.IMAGE_SIZE,
            bench=cfg.DATASET.IMAGENET9.BENCHMARK_VAL,
        )

        test_loader = get_background_challenge_data(
            root=cfg.DATASET.IMAGENET9.ROOT_IMAGENET_BG,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.IMAGENET9.IMAGE_SIZE,
            bench=cfg.DATASET.IMAGENET9.BENCHMARK_TEST,
        )

        dataset = {}
        dataset["num_class"] = 9
        dataset["num_groups"] = 9
        dataset["biases"] = [cfg.DATASET.IMAGENET9.BIAS]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
        }

        dataset["target2name"] = {
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

        dataset["root"] = cfg.DATASET.IMAGENET9.ROOT_IMAGENET_BG
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
        ):
            tag_train_loader, _ = get_imagenet9l(
                root=cfg.DATASET.IMAGENET9.ROOT_IMAGENET,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.IMAGENET9.IMAGE_SIZE,
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
                shuffle=False,
            )
            tag_test_loader = get_background_challenge_data(
                root=cfg.DATASET.IMAGENET9.ROOT_IMAGENET_BG,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.IMAGENET9.IMAGE_SIZE,
                bench="original",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            dataset["dataloaders"]["tag_train"] = tag_train_loader
            dataset["dataloaders"]["tag_test"] = tag_test_loader
    elif dataset_name == "cifar10":
        if method_name == "groupdro":
            raise ValueError(
                "GroupDro requires bias attribute annotations! The cifar10 dataset does not offer such information. Please select another method, or modify cifar10 class so that it incorporates your own bias annotations."
            )
        else:
            train_loader = get_cifar10_loaders(
                root=cfg.DATASET.CIFAR10.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                image_size=cfg.DATASET.CIFAR10.IMAGE_SIZE,
                split="train",
            )

        val_loader = get_cifar10_loaders(
            root=cfg.DATASET.CIFAR10.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.CIFAR10.IMAGE_SIZE,
            split="test",
        )

        test_loader = get_cifar10_loaders(
            root=cfg.DATASET.CIFAR10.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.CIFAR10.IMAGE_SIZE,
            split="test",
        )

        dataset = {}
        dataset["num_class"] = 10
        dataset["num_groups"] = 10
        dataset["biases"] = [cfg.DATASET.CIFAR10.BIAS]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

        dataset["sets"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

        dataset["target2name"] = {
            0: "Airplane",
            1: "Car",
            2: "Bird",
            3: "Cat",
            4: "Deer",
            5: "Dog",
            6: "Frog",
            7: "Horse",
            8: "Ship",
            9: "Truck",
        }

        dataset["root"] = cfg.DATASET.CIFAR10.ROOT
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
        ):
            tag_train_loader = get_cifar10_loaders(
                root=cfg.DATASET.CIFAR10.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.CIFAR10.IMAGE_SIZE,
                split="train",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            tag_test_loader = get_cifar10_loaders(
                root=cfg.DATASET.CIFAR10.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.CIFAR10.IMAGE_SIZE,
                split="test",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )

            dataset["dataloaders"]["tag_train"] = tag_train_loader
            dataset["dataloaders"]["tag_test"] = tag_test_loader
    elif dataset_name == "cifar100":
        if method_name == "groupdro":
            raise ValueError(
                "GroupDro requires bias attribute annotations! The cifar100 dataset does not offer such information. Please select another method, or modify cifar100 class so that it incorporates your own bias annotations."
            )
        else:
            train_loader = get_cifar100_loaders(
                root=cfg.DATASET.CIFAR100.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                image_size=cfg.DATASET.CIFAR100.IMAGE_SIZE,
                split="train",
            )

        val_loader = get_cifar100_loaders(
            root=cfg.DATASET.CIFAR100.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.CIFAR100.IMAGE_SIZE,
            split="test",
        )

        test_loader = get_cifar100_loaders(
            root=cfg.DATASET.CIFAR100.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.CIFAR100.IMAGE_SIZE,
            split="test",
        )

        dataset = {}
        dataset["num_class"] = 100
        dataset["num_groups"] = 10
        dataset["biases"] = [cfg.DATASET.CIFAR100.BIAS]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

        dataset["sets"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

        class_names = (
            train_loader.dataset.classes
        )  # This returns a list of class names in order of indices

        # Create a dictionary mapping index to class name
        dataset["target2name"] = {idx: name for idx, name in enumerate(class_names)}

        dataset["root"] = cfg.DATASET.CIFAR100.ROOT
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
        ):
            tag_train_loader = get_cifar100_loaders(
                root=cfg.DATASET.CIFAR100.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.CIFAR100.IMAGE_SIZE,
                split="train",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            tag_test_loader = get_cifar100_loaders(
                root=cfg.DATASET.CIFAR100.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.CIFAR100.IMAGE_SIZE,
                split="test",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )

            dataset["dataloaders"]["tag_train"] = tag_train_loader
            dataset["dataloaders"]["tag_test"] = tag_test_loader
    elif dataset_name == "imagenet":

        # ImageNet standard normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Transforms
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # Datasets
        train_dataset = ImageNet(
            root=f"{cfg.DATASET.IMAGENET.ROOT}/train", transform=train_transform
        )
        val_dataset = ImageNet(
            root=f"{cfg.DATASET.IMAGENET.ROOT}/imagenet-val", transform=val_transform
        )

        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        dataset = {}
        dataset["num_class"] = 1000
        dataset["num_groups"] = 10
        dataset["biases"] = [cfg.DATASET.IMAGENET.BIAS]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": val_loader,
        }

        dataset["sets"] = {
            "train": train_loader,
            "val": val_loader,
            "test": val_loader,
        }

        class_names = (
            train_loader.dataset.classes
        )  # This returns a list of class names in order of indices

        # Create a dictionary mapping index to class name
        dataset["target2name"] = {idx: name for idx, name in enumerate(class_names)}

        dataset["root"] = cfg.DATASET.IMAGENET.ROOT

    elif dataset_name == "stanford_dogs":
        if method_name == "groupdro":
            raise ValueError(
                "GroupDro requires bias attribute annotations! The stanford_dogs dataset does not offer such information. Please select another method, or modify stanford_dogs class so that it incorporates your own bias annotations."
            )
        else:
            train_loader = get_stanford_dogs_loader(
                root=cfg.DATASET.STANFORD_DOGS.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                image_size=cfg.DATASET.STANFORD_DOGS.IMAGE_SIZE,
                split="train",
            )

        val_loader = get_stanford_dogs_loader(
            root=cfg.DATASET.STANFORD_DOGS.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.STANFORD_DOGS.IMAGE_SIZE,
            split="test",
        )

        test_loader = get_stanford_dogs_loader(
            root=cfg.DATASET.STANFORD_DOGS.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.STANFORD_DOGS.IMAGE_SIZE,
            split="test",
        )

        dataset = {}
        dataset["num_class"] = 120
        dataset["num_groups"] = 120
        dataset["biases"] = [cfg.DATASET.STANFORD_DOGS.BIAS]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }

        class_names = (
            train_loader.dataset.classes
        )  # This returns a list of class names in order of indices
        class_names = [name + " dog" for name in class_names]

        # Create a dictionary mapping index to class name
        dataset["target2name"] = {idx: name for idx, name in enumerate(class_names)}

        dataset["root"] = cfg.DATASET.STANFORD_DOGS.ROOT
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
        ):
            tag_train_loader = get_stanford_dogs_loader(
                root=cfg.DATASET.STANFORD_DOGS.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.STANFORD_DOGS.IMAGE_SIZE,
                split="train",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            tag_test_loader = get_stanford_dogs_loader(
                root=cfg.DATASET.STANFORD_DOGS.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.STANFORD_DOGS.IMAGE_SIZE,
                split="test",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )

            dataset["dataloaders"]["tag_train"] = tag_train_loader
            dataset["dataloaders"]["tag_test"] = tag_test_loader
    elif dataset_name == "urbancars":
        # if method_name == "groupdro":
        #     raise ValueError("not implemented")
        # else:
        train_loader, train_dataset = get_urbancars_loader(
            root=cfg.DATASET.URBANCARS.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.URBANCARS.IMAGE_SIZE,
            split="train",
        )

        val_loader, val_dataset = get_urbancars_loader(
            root=cfg.DATASET.URBANCARS.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.URBANCARS.IMAGE_SIZE,
            split="test",
        )

        test_loader, test_dataset = get_urbancars_loader(
            root=cfg.DATASET.URBANCARS.ROOT,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            image_size=cfg.DATASET.URBANCARS.IMAGE_SIZE,
            split="test",
        )

        dataset = {}
        dataset["num_class"] = 2
        dataset["num_groups"] = 2 * 2 * 2
        dataset["biases"] = ["background", "object"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }

        # Create a dictionary mapping index to class name
        dataset["target2name"] = {0: "urban_car", 1: "country_car"}

        dataset["root"] = cfg.DATASET.URBANCARS.ROOT
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
        ):
            tag_train_loader, _ = get_urbancars_loader(
                root=cfg.DATASET.URBANCARS.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.URBANCARS.IMAGE_SIZE,
                split="train",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )
            tag_test_loader, _ = get_urbancars_loader(
                root=cfg.DATASET.URBANCARS.ROOT,
                batch_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.BATCH_SIZE,
                image_size=cfg.DATASET.URBANCARS.IMAGE_SIZE,
                split="test",
                transform=get_transform(
                    image_size=cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE
                ),
            )

            dataset["dataloaders"]["tag_train"] = tag_train_loader
            dataset["dataloaders"]["tag_test"] = tag_test_loader
    elif dataset_name == "jigsaw_toxic_comments":
        from .jigsaw_toxic_comments import get_jigsaw_toxic_comments_loaders

        if method_name == "groupdro":
            (
                train_loader,
                val_loader,
                test_loader,
                train_dataset,
                val_dataset,
                test_dataset,
            ) = get_jigsaw_toxic_comments_loaders(
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.ROOT,
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.TRAIN_SET,
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.VAL_SET,
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.TEST_SET,
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.BIAS,
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.TARGET,
                cfg.SOLVER.BATCH_SIZE,
                cfg.DATASET.NUM_WORKERS,
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.TEXT_ENCODER,
                sampler="weighted",
            )
        else:
            (
                train_loader,
                val_loader,
                test_loader,
                train_dataset,
                val_dataset,
                test_dataset,
            ) = get_jigsaw_toxic_comments_loaders(
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.ROOT,
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.TRAIN_SET,
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.VAL_SET,
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.TEST_SET,
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.BIAS,
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.TARGET,
                cfg.SOLVER.BATCH_SIZE,
                cfg.DATASET.NUM_WORKERS,
                cfg.DATASET.JIGSAW_TOXIC_COMMENTS.TEXT_ENCODER,
            )

        dataset = {}
        dataset["num_class"] = 2
        dataset["num_groups"] = 4
        dataset["biases"] = ["bias"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        dataset["root"] = cfg.DATASET.JIGSAW_TOXIC_COMMENTS.ROOT
        dataset["target2name"] = {
            0: "neutral",
            1: "identity-hate",
        }
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
            or method_name == "mhmavias"
        ):
            raise NotImplementedError(
                "this is a text dataset - cannot extract tags like images. Needs a new implementation."
            )
    elif dataset_name == "bias_in_bios":
        from my_datasets.bias_in_bios import get_bias_in_bios_loaders

        if method_name == "groupdro":
            (
                train_loader,
                val_loader,
                test_loader,
                train_dataset,
                val_dataset,
                test_dataset,
            ) = get_bias_in_bios_loaders(
                root=cfg.DATASET.BIAS_IN_BIOS.ROOT,
                target=cfg.DATASET.BIAS_IN_BIOS.TARGET,
                bias=cfg.DATASET.BIAS_IN_BIOS.BIAS,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                encoder_name=cfg.DATASET.BIAS_IN_BIOS.TEXT_ENCODER,
                sampler="weighted",
            )
        else:
            (
                train_loader,
                val_loader,
                test_loader,
                train_dataset,
                val_dataset,
                test_dataset,
            ) = get_bias_in_bios_loaders(
                root=cfg.DATASET.BIAS_IN_BIOS.ROOT,
                target=cfg.DATASET.BIAS_IN_BIOS.TARGET,
                bias=cfg.DATASET.BIAS_IN_BIOS.BIAS,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                encoder_name=cfg.DATASET.BIAS_IN_BIOS.TEXT_ENCODER,
            )

        dataset = {}
        dataset["num_class"] = 28
        dataset["num_groups"] = 28 * 2
        dataset["biases"] = ["bias"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        dataset["root"] = cfg.DATASET.BIAS_IN_BIOS.ROOT
        dataset["target2name"] = {
            0: "accountant",
            1: "architect",
            2: "attorney",
            3: "chiropractor",
            4: "comedian",
            5: "composer",
            6: "dentist",
            7: "dietitian",
            8: "dj",
            9: "filmmaker",
            10: "interior_designer",
            11: "journalist",
            12: "model",
            13: "nurse",
            14: "painter",
            15: "paralegal",
            16: "pastor",
            17: "personal_trainer",
            18: "photographer",
            19: "physician",
            20: "poet",
            21: "professor",
            22: "psychologist",
            23: "rapper",
            24: "software_engineer",
            25: "surgeon",
            26: "teacher",
            27: "yoga_teacher",
        }
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
            or method_name == "mhmavias"
        ):
            raise NotImplementedError(
                "this is a text dataset - cannot extract tags like images. Needs a new implementation."
            )
    elif dataset_name == "speech_accent_archive":
        if method_name == "groupdro":
            (
                train_loader,
                val_loader,
                test_loader,
                train_dataset,
                val_dataset,
                test_dataset,
            ) = get_speech_accent_dataloaders(
                root=cfg.DATASET.SPEECH_ACCENT_ARCHIVE.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                encoder_name=cfg.DATASET.SPEECH_ACCENT_ARCHIVE.AUDIO_ENCODER,
                sampler="weighted",
            )
        else:
            (
                train_loader,
                val_loader,
                test_loader,
                train_dataset,
                val_dataset,
                test_dataset,
            ) = get_speech_accent_dataloaders(
                root=cfg.DATASET.SPEECH_ACCENT_ARCHIVE.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                encoder_name=cfg.DATASET.SPEECH_ACCENT_ARCHIVE.AUDIO_ENCODER,
            )

        dataset = {}
        dataset["num_class"] = 2
        dataset["num_groups"] = 2 * 2
        dataset["biases"] = ["bias"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        dataset["root"] = cfg.DATASET.SPEECH_ACCENT_ARCHIVE.ROOT
        dataset["target2name"] = {
            0: "non_english",
            1: "english",
        }
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
            or method_name == "mhmavias"
        ):
            raise NotImplementedError(
                "this is a text dataset - cannot extract tags like images. Needs a new implementation."
            )
    elif dataset_name == "urbansounds":
        if method_name == "groupdro":
            (
                train_loader,
                val_loader,
                test_loader,
                train_dataset,
                val_dataset,
                test_dataset,
            ) = get_urbansounds_dataloaders(
                root=cfg.DATASET.URBANSOUNDS.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                encoder_name=cfg.DATASET.URBANSOUNDS.AUDIO_ENCODER,
                sampler="weighted",
            )
        else:
            (
                train_loader,
                val_loader,
                test_loader,
                train_dataset,
                val_dataset,
                test_dataset,
            ) = get_urbansounds_dataloaders(
                root=cfg.DATASET.URBANSOUNDS.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                encoder_name=cfg.DATASET.URBANSOUNDS.AUDIO_ENCODER,
            )

        dataset = {}
        dataset["num_class"] = 10
        dataset["num_groups"] = 10 * 2
        dataset["biases"] = ["bias"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        dataset["root"] = cfg.DATASET.URBANSOUNDS.ROOT
        dataset["target2name"] = {
            0: "air_conditioner",
            1: "car_horn",
            2: "children_playing",
            3: "dog_bark",
            4: "drilling",
            5: "engine_idling",
            6: "gun_shot",
            7: "jackhammer",
            8: "siren",
            9: "street_music",
        }
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
            or method_name == "mhmavias"
        ):
            raise NotImplementedError(
                "this is a text dataset - cannot extract tags like images. Needs a new implementation."
            )
    elif dataset_name == "urbansounds58":
        if method_name == "groupdro":
            (
                train_loader,
                val_loader,
                test_loader,
                train_dataset,
                val_dataset,
                test_dataset,
            ) = get_urbansounds58_dataloaders(
                root=cfg.DATASET.URBANSOUNDS.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                encoder_name=cfg.DATASET.URBANSOUNDS.AUDIO_ENCODER,
                sampler="weighted",
            )
        else:
            (
                train_loader,
                val_loader,
                test_loader,
                train_dataset,
                val_dataset,
                test_dataset,
            ) = get_urbansounds58_dataloaders(
                root=cfg.DATASET.URBANSOUNDS.ROOT,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                encoder_name=cfg.DATASET.URBANSOUNDS.AUDIO_ENCODER,
            )

        dataset = {}
        dataset["num_class"] = 2
        dataset["num_groups"] = 2 * 2
        dataset["biases"] = ["bias"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        dataset["root"] = cfg.DATASET.URBANSOUNDS.ROOT
        dataset["target2name"] = {
            0: "engine_idling",
            1: "siren",
        }
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
            or method_name == "mhmavias"
        ):
            raise NotImplementedError(
                "this is a text dataset - cannot extract tags like images. Needs a new implementation."
            )
    elif dataset_name == "chexpert_nih":
        if method_name == "groupdro":
            (
                train_loader,
                train_dataset,
                val_loader,
                val_dataset,
                test_loader,
                test_dataset,
            ) = get_chexpert_nih_loader(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                sampler="weighted",
            )
        else:
            (
                train_loader,
                train_dataset,
                val_loader,
                val_dataset,
                test_loader,
                test_dataset,
            ) = get_chexpert_nih_loader(
                batch_size=cfg.SOLVER.BATCH_SIZE,
            )

        dataset = {}
        dataset["num_class"] = 2
        dataset["num_groups"] = 2 * 2
        dataset["biases"] = ["bias"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        dataset["root"] = None
        dataset["target2name"] = {
            0: "negative",
            1: "positive",
        }
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
            or method_name == "mhmavias"
        ):
            raise NotImplementedError(
                "this is a text dataset - cannot extract tags like images. Needs a new implementation."
            )
    elif dataset_name == "ucf101":
        if cfg.MODEL.TYPE == "swin_t3d":
            transforms = (
                torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1.transforms()
            )
        else:
            transforms = None
        if method_name == "groupdro":

            train_loader, train_dataset = get_ucf101(
                cfg.DATASET.UCF101.VIDEO_PATH,
                cfg.DATASET.UCF101.ANNOTATION_PATH,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                split="train",
                sampler="weighted",
                bias_type=cfg.DATASET.UCF101.BIAS_TYPE,
                bias_th=cfg.DATASET.UCF101.BIAS_TH,
                transform=transforms,
                sample_duration=cfg.DATASET.UCF101.SAMPLE_DURATION
            )
        else:
            train_loader, train_dataset = get_ucf101(
                cfg.DATASET.UCF101.VIDEO_PATH,
                cfg.DATASET.UCF101.ANNOTATION_PATH,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                split="train",
                bias_type=cfg.DATASET.UCF101.BIAS_TYPE,
                bias_th=cfg.DATASET.UCF101.BIAS_TH,
                transform=transforms,
                sample_duration=cfg.DATASET.UCF101.SAMPLE_DURATION
            )

        val_loader, val_dataset = get_ucf101(
            cfg.DATASET.UCF101.VIDEO_PATH,
            cfg.DATASET.UCF101.ANNOTATION_PATH,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            split="val",
            bias_type=cfg.DATASET.UCF101.BIAS_TYPE,
            bias_th=cfg.DATASET.UCF101.BIAS_TH,
            transform=transforms,
            sample_duration=cfg.DATASET.UCF101.SAMPLE_DURATION
        )
        if cfg.DATASET.UCF101.TEST_BENCHMARK == "original":
            test_loader, test_dataset = get_ucf101(
                cfg.DATASET.UCF101.VIDEO_PATH,
                cfg.DATASET.UCF101.ANNOTATION_PATH,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                split="test",
                bias_type=cfg.DATASET.UCF101.BIAS_TYPE,
                bias_th=cfg.DATASET.UCF101.BIAS_TH,
                transform=transforms,
                sample_duration=cfg.DATASET.UCF101.SAMPLE_DURATION
            )
        elif cfg.DATASET.UCF101.TEST_BENCHMARK == "scuba":
            test_loader, test_dataset = get_ucf101(
                cfg.DATASET.UCF101.VIDEO_PATH_SCUBA,
                cfg.DATASET.UCF101.ANNOTATION_PATH_SCUBA,
                batch_size=cfg.SOLVER.BATCH_SIZE,
                split="test",
                bias_type=cfg.DATASET.UCF101.BIAS_TYPE,
                bias_th=cfg.DATASET.UCF101.BIAS_TH,
                version=cfg.DATASET.UCF101.TEST_BENCHMARK,
                transform=transforms,
                sample_duration=cfg.DATASET.UCF101.SAMPLE_DURATION
            )
            # test_loader, test_dataset = get_ucf101(
            #     cfg.DATASET.UCF101.VIDEO_PATH,
            #     cfg.DATASET.UCF101.ANNOTATION_PATH,
            #     batch_size=cfg.SOLVER.BATCH_SIZE,
            #     split="test",
            #     bias_type=cfg.DATASET.UCF101.BIAS_TYPE,
            #     bias_th=cfg.DATASET.UCF101.BIAS_TH,
            # )
        dataset = {}
        dataset["num_class"] = 101
        dataset["num_groups"] = 101 * 14
        dataset["biases"] = ["bias"]
        dataset["dataloaders"] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        dataset["sets"] = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        dataset["root"] = None
        dataset["target2name"] = {
            0: "ApplyEyeMakeup",
            1: "ApplyLipstick",
            2: "Archery",
            3: "BabyCrawling",
            4: "BalanceBeam",
            5: "BandMarching",
            6: "BaseballPitch",
            7: "Basketball",
            8: "BasketballDunk",
            9: "BenchPress",
            10: "Biking",
            11: "Billiards",
            12: "BlowDryHair",
            13: "BlowingCandles",
            14: "BodyWeightSquats",
            15: "Bowling",
            16: "BoxingPunchingBag",
            17: "BoxingSpeedBag",
            18: "BreastStroke",
            19: "BrushingTeeth",
            20: "CleanAndJerk",
            21: "CliffDiving",
            22: "CricketBowling",
            23: "CricketShot",
            24: "CuttingInKitchen",
            25: "Diving",
            26: "Drumming",
            27: "Fencing",
            28: "FieldHockeyPenalty",
            29: "FloorGymnastics",
            30: "FrisbeeCatch",
            31: "FrontCrawl",
            32: "GolfSwing",
            33: "Haircut",
            34: "Hammering",
            35: "HammerThrow",
            36: "HandstandPushups",
            37: "HandstandWalking",
            38: "HeadMassage",
            39: "HighJump",
            40: "HorseRace",
            41: "HorseRiding",
            42: "HulaHoop",
            43: "IceDancing",
            44: "JavelinThrow",
            45: "JugglingBalls",
            46: "JumpingJack",
            47: "JumpRope",
            48: "Kayaking",
            49: "Knitting",
            50: "LongJump",
            51: "Lunges",
            52: "MilitaryParade",
            53: "Mixing",
            54: "MoppingFloor",
            55: "Nunchucks",
            56: "ParallelBars",
            57: "PizzaTossing",
            58: "PlayingCello",
            59: "PlayingDaf",
            60: "PlayingDhol",
            61: "PlayingFlute",
            62: "PlayingGuitar",
            63: "PlayingPiano",
            64: "PlayingSitar",
            65: "PlayingTabla",
            66: "PlayingViolin",
            67: "PoleVault",
            68: "PommelHorse",
            69: "PullUps",
            70: "Punch",
            71: "PushUps",
            72: "Rafting",
            73: "RockClimbingIndoor",
            74: "RopeClimbing",
            75: "Rowing",
            76: "SalsaSpin",
            77: "ShavingBeard",
            78: "Shotput",
            79: "SkateBoarding",
            80: "Skiing",
            81: "Skijet",
            82: "SkyDiving",
            83: "SoccerJuggling",
            84: "SoccerPenalty",
            85: "StillRings",
            86: "SumoWrestling",
            87: "Surfing",
            88: "Swing",
            89: "TableTennisShot",
            90: "TaiChi",
            91: "TennisSwing",
            92: "ThrowDiscus",
            93: "TrampolineJumping",
            94: "Typing",
            95: "UnevenBars",
            96: "VolleyballSpiking",
            97: "WalkingWithDog",
            98: "WallPushups",
            99: "WritingOnBoard",
            100: "YoYo",
        }
        if (
            method_name == "mavias"
            or method_name == "erm_tags"
            or metric_name == "wg_ovr_tags"
            or method_name == "mhmavias"
        ):
            raise NotImplementedError(
                "this is a text dataset - cannot extract tags like images. Needs a new implementation."
            )
    return dataset
