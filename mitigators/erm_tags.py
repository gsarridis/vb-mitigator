"""
Module for ERMTagsTrainer class and related functions.
"""

import os
import json
import re
import ast

import numpy as np
import ollama
from tqdm import tqdm
import pandas as pd
import torch
from ram.models import ram_plus

from models.builder import get_model
from .base_trainer import BaseTrainer
from tools.utils import load_ollama_docker
from tools.metrics import metrics_dicts, get_performance
from tools.utils import log_msg


# Function to check if a sample has any of the saved tags for its class
def has_saved_tag(sample_tags, class_name, tags_above_threshold):
    saved_tags = tags_above_threshold.get(class_name, [])
    # Check if there is any intersection between sample tags and saved tags
    return any(tag in saved_tags for tag in sample_tags)


class ERMTagsTrainer(BaseTrainer):

    def _setup_models(self):
        self.model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
            pretrained=self.cfg.MODEL.PRETRAINED,
        )
        self.model.to(self.device)

    def get_overperforming_tags(self, stage="test", threshold=None, epoch=0):
        self._set_eval()
        # threshold = threshold / 100
        with torch.no_grad():
            all_data = {key: [] for key in self.biases}
            all_data["targets"] = []
            all_data["predictions"] = []

            # Initialize dictionaries to track tag accuracies
            class_tag_correct_counts = {
                class_name: {} for class_name in self.target2name.values()
            }
            class_tag_total_counts = {
                class_name: {} for class_name in self.target2name.values()
            }

            for batch in self.dataloaders[stage]:
                batch_dict, loss = self._val_iter(batch)
                predictions = batch_dict["predictions"]
                targets = batch_dict["targets"].cpu().numpy()
                indices = batch["index"]

                for i in range(len(indices)):
                    class_name = self.target2name[targets[i]]
                    irrelevant_tags = self.index_to_tags_test[indices[i].item()]
                    irrelevant_tags = [
                        tag.strip() for tag in irrelevant_tags.split(", ")
                    ]

                    correct = int(predictions[i] == targets[i])

                    for tag in irrelevant_tags:
                        if tag not in class_tag_correct_counts[class_name]:
                            class_tag_correct_counts[class_name][tag] = correct
                            class_tag_total_counts[class_name][tag] = 1
                        else:
                            class_tag_correct_counts[class_name][tag] += correct
                            class_tag_total_counts[class_name][tag] += 1

            # Compute accuracy for each tag and filter those above the threshold

            tags_above_threshold = {
                class_name: [] for class_name in self.target2name.values()
            }
            tags_above_threshold_acc = {
                class_name: [] for class_name in self.target2name.values()
            }
            tags_above_threshold_samples = {
                class_name: [] for class_name in self.target2name.values()
            }
            name2idx = {}
            for key, value in self.target2name.items():
                name2idx[value] = key

            for class_name in class_tag_correct_counts:
                for tag in class_tag_correct_counts[class_name]:
                    accuracy = (
                        class_tag_correct_counts[class_name][tag]
                        / class_tag_total_counts[class_name][tag]
                    )
                    if accuracy > (threshold[f"accuracy_{name2idx[class_name]}"] / 100):
                        tags_above_threshold[class_name].append(tag)
                        tags_above_threshold_acc[class_name].append(accuracy)
                        tags_above_threshold_samples[class_name].append(
                            class_tag_total_counts[class_name][tag]
                        )

            # Save results to CSV and JSON
            tags_for_csv = []
            for class_name, tags, accs, n_samples in zip(
                tags_above_threshold.keys(),
                tags_above_threshold.values(),
                tags_above_threshold_acc.values(),
                tags_above_threshold_samples.values(),
            ):
                for tag, acc, n_s in zip(tags, accs, n_samples):
                    tags_for_csv.append([class_name, tag, acc, n_s])

            tags_df = pd.DataFrame(
                tags_for_csv, columns=["Class", "Tag", "Acc", "Samples"]
            )
            tags_df = tags_df.sort_values(by="Acc", ascending=False)
            ov_name = f"overperforming_tags_ep{epoch}.csv"
            tags_df.to_csv(
                os.path.join(self.data_root, ov_name),
                index=False,
            )
            return tags_df

    def train(self):
        start_epoch = self.current_epoch
        for epoch in range(
            start_epoch,
            min(start_epoch + self.cfg.EXPERIMENT.EPOCH_STEPS, self.cfg.SOLVER.EPOCHS),
        ):
            self.current_epoch = epoch
            log_dict = self._train_epoch()
            # log_dict = {}
            if self.cfg.LOG.TRAIN_PERFORMANCE:
                train_performance = self._validate_epoch(stage="train")
                train_log_dict = self.build_log_dict(train_performance, stage="train")
                log_dict.update(train_log_dict)
            if self.cfg.LOG.SAVE_CRITERION == "val":
                val_performance = self._validate_epoch(stage="val")
                val_log_dict = self.build_log_dict(val_performance, stage="val")
                log_dict.update(val_log_dict)
            test_performance = self._validate_epoch(stage="test")
            test_log_dict = self.build_log_dict(test_performance, stage="test")
            log_dict.update(test_log_dict)
            update_cpkt = self._update_best(log_dict)
            if update_cpkt:
                self._save_checkpoint(tag="best")
            self.overperforming_tags_df = self.get_overperforming_tags(
                stage="test", threshold=test_performance, epoch=epoch
            )
            test_performance_tags = self._validate_epoch_tags(stage="test")
            log_dict.update(
                self.build_log_dict(test_performance_tags, stage="test_tags")
            )
            self._log_epoch(log_dict, update_cpkt)
            self._save_checkpoint(tag=f"current_{self.cfg.EXPERIMENT.SEED}")

        self._save_checkpoint(tag="latest")

    def _method_specific_setups(self):
        self.tags_df_test = self.get_ram_tags(split="test")
        self.get_relevant_tags()
        self.tags_df_test = self.get_irrelevant_tags(self.tags_df_test, split="test")
        self.index_to_tags_test = {
            row["index"]: (
                row["irrelevant_tags"].replace(" | ", ", ")
                if isinstance(row["irrelevant_tags"], str)
                else " "
            )
            for _, row in self.tags_df_test.iterrows()
        }
        return

    def get_irrelevant_tags(self, df, split="train"):
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
        outdir = self.data_root
        # Check if the file exists
        p = os.path.join(outdir, f"{split}_tags.csv")
        if os.path.isfile(os.path.join(outdir, f"{split}_tags.csv")):
            print(f"Loading tags from {p}")
            # Load the tags from the CSV file
            split_tags_df = pd.read_csv(os.path.join(outdir, f"{split}_tags.csv"))
        else:
            print(f"Extracting tags and saving to {p}")
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

            # Loop through batches
            for i, batch in enumerate(tqdm(data_loader)):
                # Initialize an empty list to store tags
                tag_list = []
                index_list = []
                target_list = []

                images = batch["inputs"].to(self.device)
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

    def eval(self):
        self.load_checkpoint(self.cfg.MODEL.PATH)
        test_performance = self._validate_epoch(stage="test")
        test_log_dict = self.build_log_dict(test_performance, stage="test")
        self.overperforming_tags_df = self.get_overperforming_tags(
            stage="test", threshold=test_performance
        )
        test_performance_tags = self._validate_epoch_tags(stage="test")
        test_log_dict.update(
            self.build_log_dict(test_performance_tags, stage="test_tags")
        )
        print(log_msg(f"Test performance: {test_log_dict}", "EVAL", self.logger))
