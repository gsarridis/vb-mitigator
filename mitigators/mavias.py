"""
Module for MAVIASTrainer class and related functions.
"""

import os
import json
import re
import ast
import sys

from matplotlib import pyplot as plt
import numpy as np
import ollama
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
from ram.models import ram_plus

from models.builder import get_model
from models.simple_mlp import SimpleMLP
from .base_trainer import BaseTrainer
from tools.utils import load_ollama_docker
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage


class MAVIASTrainer(BaseTrainer):

    def _setup_models(self):
        self.model = get_model(
            self.cfg.MODEL.TYPE, self.num_class, pretrained=self.cfg.MODEL.PRETRAINED
        )
        if self.cfg.MODEL.FREEZE_BACKBONE:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

        self.model.to(self.device)

        self.proj_net = SimpleMLP(
            self.cfg.MITIGATOR.MAVIAS.ENCODER.SIZE, self.model.fc.in_features
        )
        self.proj_net.to(self.device)

        clip_models = [
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
        ]

        clip_model_id = clip_models[2]

        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_id)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_id)
        self.text_encoder.to(self.device)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id).to(self.device)

    def _setup_optimizer(self):
        super()._setup_optimizer()
        parameters_projection = self.proj_net.parameters()

        if self.cfg.MITIGATOR.MAVIAS.PROJNET.OPTIM.TYPE == "SGD":
            self.optimizer_projection = torch.optim.SGD(
                parameters_projection,
                lr=self.cfg.MITIGATOR.MAVIAS.PROJNET.OPTIM.LR,
                momentum=self.cfg.MITIGATOR.MAVIAS.PROJNET.OPTIM.MOMENTUM,
                weight_decay=self.cfg.MITIGATOR.MAVIAS.PROJNET.OPTIM.WEIGHT_DECAY,
            )
        elif self.cfg.MITIGATOR.MAVIAS.PROJNET.OPTIM.TYPE == "Adam":
            self.optimizer_projection = torch.optim.Adam(
                parameters_projection,
                lr=self.cfg.MITIGATOR.MAVIAS.PROJNET.OPTIM.LR,
                weight_decay=self.cfg.MITIGATOR.MAVIAS.PROJNET.OPTIM.WEIGHT_DECAY,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.cfg.MITIGATOR.MAVIAS.PROJNET.OPTIM.TYPE}"
            )

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        indices = batch["index"]

        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_projection.zero_grad(set_to_none=True)

        tags = [self.index_to_tags[index.item()] for index in indices]
        tag_emb = torch.stack([self.precomputed_embeddings[tag] for tag in tags]).to(
            self.device
        )

        b_feats = self.proj_net(tag_emb)

        logits, logits2 = self.model.mavias_forward(inputs, b_feats)
        tmp = logits2.detach().cpu().clone()
        norm_main = torch.norm(logits)
        norm_clip = torch.norm(tmp).to(self.device)
        norm_loss = F.mse_loss(
            norm_main, norm_clip * self.cfg.MITIGATOR.MAVIAS.LOSS.LAMBDA
        )
        ce_loss = self.criterion(logits + logits2, targets)

        loss = ce_loss + self.cfg.MITIGATOR.MAVIAS.LOSS.ALPHA * norm_loss

        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()

        return {"train_cls_loss": ce_loss, "train_norm_loss": norm_loss}

    def _optimizer_step(self):
        self.optimizer.step()
        self.optimizer_projection.step()

    def _set_train(self):
        self.proj_net.train()
        return super()._set_train()

    def _set_eval(self):
        self.proj_net.eval()
        return super()._set_eval()

    def _method_specific_setups(self):
        self.tags_df = self.get_ram_tags(split="train")
        self.get_relevant_tags()
        self.tags_df = self.get_irrelevant_tags(self.tags_df, split="train")
        # print(self.tags_df)
        self.index_to_tags = {
            row["index"]: (
                row["irrelevant_tags"].replace(" | ", ", ")
                if isinstance(row["irrelevant_tags"], str)
                else " "
            )
            for _, row in self.tags_df.iterrows()
        }
        self.precomputed_embeddings = self.precompute_text_embeddings(split="train")

    def get_irrelevant_tags(self, df, split="train"):
        # check if self.tags_df["irrelevant_tags"] exists
        # if "irrelevant_tags" in self.tags_df.columns:
        #     return
        # else:

        def calc_irrelevant_tags(row):
            tags = str(row["tags"]) if isinstance(row["tags"], str) else ""
            all_tags = set(tag.strip() for tag in tags.split(" | "))
            relevant = set(self.relevant_tags.get(row["target"], []))
            return " | ".join(all_tags - relevant)

        df["irrelevant_tags"] = df.apply(calc_irrelevant_tags, axis=1)
        # save to csv
        df.to_csv(os.path.join(self.data_root, f"{split}_tags.csv"), index=False)
        return df

    def get_relevant_tags(self):
        llm_name = self.cfg.MITIGATOR.MAVIAS.LLM.TYPE
        batch_size = self.cfg.MITIGATOR.MAVIAS.LLM.BATCH_SIZE

        # Initialize an empty list to store the tags
        tags = []
        # Open the CSV file and read the tags
        unique_tags_df = pd.read_csv(
            os.path.join(self.data_root, "unique_tags_per_class.csv")
        )

        unique_tags_df["tags"] = unique_tags_df["tags"].apply(ast.literal_eval)

        self.relevant_tags = {}
        flag = False
        for target in unique_tags_df["class"]:
            path_to_check = os.path.join(
                self.data_root,
                "relevant_tags",
                f"{self.cfg.MITIGATOR.MAVIAS.LLM.TYPE}_bs{batch_size}_{target}_{self.target2name[target]}.csv",
            )
            # if path existis
            if not os.path.exists(path_to_check):
                flag = True
                break
            else:
                print(f"Loading relevant tags from {path_to_check}")
                self.relevant_tags[target] = pd.read_csv(path_to_check)["tags"].tolist()
        if flag:
            # Check if the ollama executable exists
            load_ollama_docker(llm_name)

            # Function to split list into batches of 100
            def split_into_batches(lst, batch_size):
                for i in range(0, len(lst), batch_size):
                    yield lst[i : i + batch_size]

            for target, tags in zip(unique_tags_df["class"], unique_tags_df["tags"]):
                # print(f"tags: {tags[0]}")
                # Initialize an empty list to collect relevant tags
                relevant_tags = []
                # Process tags in batches of 100

                for batch in split_into_batches(tags, batch_size):
                    print(f"batch: {batch}")
                    tag_list = ", ".join(
                        batch
                    )  # Join the batch into a comma-separated string
                    # print(f"input tag list: {tag_list}")
                    response = ollama.chat(
                        model=llm_name,
                        messages=[
                            {
                                "role": "system",
                                "content": """
                I will provide you with the name of a target class and a large list of tags. Your task is to evaluate the tags and identify only those directly related to the target class. A tag is considered relevant if it describes or is an essential part of the object associated with the class name. This includes tags that refer to:
                physical components, defining features, inherent characteristics, and essential behaviors or functions of the object.
                For example, if the target class is "bee," tags like "insect," "wing," and "buzz," are relevant because they describe core aspects of what a bee is or does.

                Conversely, a tag is irrelevant if it refers to elements that are not an intrinsic part of the object. Irrelevant tags may include: 
                background details, environmental context, colors (unless a defining characteristic), lighting, textures, other objects, or other non-essential contextual elements.
                For example, in the case of the class "bee," tags like "sky," "flower," or "blue" would be irrelevant, as they describe the environment or background rather than the bee itself.

                Also, note that when it comes to humans, attributes like gender, race, age, religion, or other personal characteristics are considered irrelevant unless they are essential to identify the target class. 

                Please output only the relevant tags in JSON format only (i.e., {
                relevant_tags: [
                the list of tags
                ] 
                }).
                                """,
                            },
                            {
                                "role": "user",
                                "content": f"""Target class: {self.target2name[target]}
                Tags: {tag_list}
                                """,
                            },
                        ],
                    )

                    # Output the result from the LLM for each batch
                    # print(response["message"]["content"])
                    output = response["message"]["content"]
                    # Use regex to find all relevant tags
                    # Match anything within braces `{}` and square brackets `[]` including strings
                    # Use regex to find all relevant tags
                    # Find all JSON objects in the text using regex
                    json_objects = re.findall(r"{[^{}]*}", output)

                    # Extracting relevant tags from each JSON object
                    for json_obj in json_objects:
                        try:
                            # Load the JSON data
                            data = json.loads(json_obj)

                            # Append the relevant tags to the list
                            relevant_tags.extend(data.get("relevant_tags", []))
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")

                # Remove duplicates and sort the list if needed
                relevant_tags = list(set(relevant_tags))
                relevant_tags.sort()

                # Output the relevant tags
                print(f"class:{target}, relevant tags: {relevant_tags}")
                # Create a DataFrame from the list of relevant tags
                df = pd.DataFrame(relevant_tags, columns=["tags"])

                # Save the DataFrame to a CSV file in the same directory
                csv_file_path = os.path.join(
                    self.data_root,
                    "relevant_tags",
                    f"{self.cfg.MITIGATOR.MAVIAS.LLM.TYPE}_bs{batch_size}_{target}_{self.target2name[target]}.csv",
                )
                os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

                df.to_csv(csv_file_path, index=False)
                self.relevant_tags[target] = relevant_tags

    def get_ram_tags(self, split="train"):

        device = self.device
        data_loader = self.dataloaders[f"tag_{split}"]
        total_samples = len(data_loader.dataset)
        outdir = self.data_root
        # Check if the file exists
        p = os.path.join(outdir, f"{split}_tags.csv")

        # Check if the CSV file exists and find the last processed index
        if os.path.isfile(p):
            print(f"Loading tags from {p}")
            split_tags_df = pd.read_csv(p)

            if not split_tags_df.empty:
                last_idx = split_tags_df["index"].max()  # Get the last processed index
            else:
                last_idx = -1
        else:
            last_idx = -1  # No file, start from the beginning

        if last_idx < total_samples - 1:
            print(f"Extracting tags from index {last_idx + 1} and saving to {p}")
            #######load model
            model = ram_plus(
                # pretrained="https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth",
                pretrained="./pretrained/ram_plus_swin_large_14m.pth",
                image_size=self.cfg.MITIGATOR.MAVIAS.TAGGING_MODEL.IMG_SIZE,
                vit="swin_l",
            )
            model.eval()

            model = model.to(device)

            # Initialize a dictionary to store unique tags for each class
            unique_tags_per_class = {}

            computed_samples = 0
            # Loop through batches
            for i, batch in enumerate(tqdm(data_loader)):
                indices = batch["index"].detach().cpu().numpy()

                # Skip batches that were already processed
                if indices[0] <= last_idx:
                    continue
                else:
                    computed_samples += len(indices)
                # Initialize an empty list to store tags
                tag_list = []
                index_list = []
                target_list = []

                images = batch["inputs"]

                # --- Check if the input is a video (5D: B, C, T, H, W) ---
                if images.ndim == 5:
                    # Keep only the first frame (T=0)
                    images = images[:, :, 0, :, :]

                # Move to device
                images = images.to(self.device)
                labels = batch["targets"]
                indices = batch["index"]

                with torch.no_grad():
                    tags, _ = model.generate_tag(images)

                # Iterate over tags and labels
                for tag_uni, label in zip(tags, labels):
                    label = label.item()  # Convert tensor to scalar
                    if label not in unique_tags_per_class:
                        unique_tags_per_class[label] = set()
                    for tag in tag_uni.split(" | "):
                        if tag != "":
                            unique_tags_per_class[label].update([tag])
                # Append tags to the tag_list
                tag_list.extend(tags)
                index_list.extend(indices.detach().cpu().numpy())
                target_list.extend(labels.detach().cpu().numpy())
                # Convert indices to numpy array and extend the list

                # Write tags to file after every batch
                tag_df = pd.DataFrame(
                    {"index": index_list, "target": target_list, "tags": tag_list}
                )
                # Append to file in each iteration
                if i == 0:
                    tag_df.to_csv(
                        os.path.join(outdir, f"{split}_tags.csv"), index=False
                    )
                else:
                    tag_df.to_csv(
                        os.path.join(outdir, f"{split}_tags.csv"),
                        mode="a",
                        index=False,
                        header=False,
                    )
                if computed_samples >= self.cfg.EXPERIMENT.PLACEHOLDER_STEPS:
                    print("Reached the maximum number of tag steps. Exiting...")
                    sys.exit(0)  # Exit with success
            split_tags_df = pd.read_csv(os.path.join(outdir, f"{split}_tags.csv"))

            if split == "train":
                # Convert sets to lists and save unique tags per class to CSV
                unique_tags_df = pd.DataFrame(
                    [
                        (label, list(tags))
                        for label, tags in unique_tags_per_class.items()
                    ],
                    columns=["class", "tags"],
                )
                unique_tags_df.to_csv(
                    os.path.join(outdir, "unique_tags_per_class.csv"), index=False
                )
        return split_tags_df

    def precompute_text_embeddings(self, split):
        # check if they are saved
        # if os.path.isfile(os.path.join(self.data_root, "clip_embeddings.pt")):
        #     print(
        #         f"Loading text embeddings from {os.path.join(self.data_root, 'clip_embeddings.pt')}"
        #     )
        #     precomputed_embeddings = torch.load(
        #         os.path.join(self.data_root, "clip_embeddings.pt")
        #     )
        #     return precomputed_embeddings
        # else:
        print("Precomputing text embeddings...")
        precomputed_embeddings = {}
        with torch.no_grad():
            for batch in tqdm(self.dataloaders[split]):

                indices = batch["index"]
                if split == "train":
                    tags = [self.index_to_tags[index.item()] for index in indices]
                else:
                    tags = [self.index_to_tags_test[index.item()] for index in indices]

                comma_separated_tags = [
                    f"a photo with {tags[i]}" for i in range(len(tags))
                ]

                text_inputs = self.tokenizer(
                    comma_separated_tags,
                    padding="max_length",
                    return_tensors="pt",
                ).to(self.device)
                b_feats = self.clip_model.get_text_features(**text_inputs)
                for prompt, emb in zip(comma_separated_tags, b_feats):
                    precomputed_embeddings[prompt.replace("a photo with ", "")] = (
                        emb.detach().cpu()
                    )

            torch.save(
                precomputed_embeddings,
                os.path.join(self.data_root, f"clip_embeddings_{split}.pt"),
            )
        return precomputed_embeddings
