import argparse
import pandas as pd
import torch.backends.cudnn as cudnn
from configs.cfg import CFG as cfg
from torchvision.models import resnet50
from torch import nn
import torch
from datasets.imagenet9 import get_background_challenge_data
from tools.metrics import get_performance
import numpy as np
from models.builder import get_model

cudnn.benchmark = True


def val_iter(batch, model):
    batch_dict = {}
    inputs = batch["inputs"].to("cuda:0")
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        outputs, _ = outputs

    batch_dict["predictions"] = torch.argmax(outputs, dim=1)
    batch_dict["targets"] = batch["targets"]

    return batch_dict


def validate_epoch(model, dataloader):
    model.eval()
    all_data = {}
    with torch.no_grad():
        all_data["targets"] = []
        all_data["predictions"] = []

        for batch in dataloader:
            batch_dict = val_iter(batch, model)
            for key, value in batch_dict.items():
                all_data[key].append(value.detach().cpu().numpy())

        for key in all_data:
            all_data[key] = np.concatenate(all_data[key])
        # metric specific data
        performance = get_performance["acc"](all_data)
    return performance


def main(cfg, method, seed):
    device = "cuda:0"
    # load model
    model_root = f"./output/imagenet9_baselines/dev/{method}/best_{seed}"

    model = get_model(
        "resnet50",
        9,
        pretrained=False,
    )
    loaded_dict = torch.load(model_root)
    model.load_state_dict(loaded_dict["model"])

    model = model.to(device)

    for bench in [
        "original",
        "mixed_rand",
        "mixed_next",
        "mixed_same",
        "no_fg",
        "only_bg_b",
        "only_bg_t",
        "only_fg",
    ]:
        test_loader = get_background_challenge_data(
            root=cfg.DATASET.IMAGENET9.ROOT_IMAGENET_BG,
            batch_size=1,
            image_size=cfg.DATASET.IMAGENET9.IMAGE_SIZE,
            bench=bench,
        )

        performance = validate_epoch(model, test_loader)
        acc = performance["accuracy"]
        print(f"{bench}: {acc}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("training")
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    cfg.freeze()
    for m in [
        "flacb",
        "softcon",
        "sd",
    ]:  # "mavias", "jtt", "lff", "erm", "debian"]:
        for s in [0, 1, 2]:
            print(f"results for {m} with seed {s}")
            main(cfg, m, s)
