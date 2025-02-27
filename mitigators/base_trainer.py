import ast
import os

import pandas as pd

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from datasets.builder import get_dataset
from models.builder import get_model
from tools.metrics import metrics_dicts, get_performance
from tools.metrics.utils import AverageMeter
from tools.utils import (
    log_msg,
    save_checkpoint,
    load_checkpoint,
    seed_everything,
    setup_logger,
)
from configs.cfg import show_cfg
from ram.models import ram_plus


class BaseTrainer:
    """
    Base class for training models.
    Args:
        cfg (Config): Configuration object containing experiment settings.
    Attributes:
        cfg (Config): Configuration object.
        logger (Logger): Logger for logging information.
        device (torch.device): Device to run the model on (CPU or GPU).
        current_epoch (int): Current epoch number.
        num_class (int): Number of classes in the dataset.
        biases (list): List of biases in the dataset.
        dataloaders (dict): Dictionary of dataloaders for training, validation, and testing.
        data_root (str): Root directory of the dataset.
        target2name (dict): Mapping from target indices to names.
        ba_groups (list): List of bias-aligned groups in the dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        criterion (torch.nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        metric_dict (dict): Dictionary containing metric information.
        best_performance (float): Best performance metric value.
        model (torch.nn.Module): Model to be trained.
        log_path (str): Path to the log directory.
        tf_writer (SummaryWriter): TensorBoard writer for logging.
    Methods:
        _setup_device(): Sets up the device for training (CPU or GPU).
        _metric_specific_setups(): Sets up metric-specific configurations.
        _setup_dataset(): Sets up the dataset and dataloaders.
        _setup_optimizer(): Sets up the optimizer.
        _setup_criterion(): Sets up the loss function.
        _loss_backward(loss): Performs backpropagation on the loss.
        _optimizer_step(): Performs an optimization step.
        _set_train(): Sets the model to training mode.
        _set_eval(): Sets the model to evaluation mode.
        _setup_scheduler(): Sets up the learning rate scheduler.
        _train_iter(batch): Performs a single training iteration.
        _train_epoch(): Performs a single training epoch.
        _val_iter(batch): Performs a single validation iteration.
        _validate_epoch(stage): Performs a single validation epoch.
        _method_specific_setups(): Placeholder for method-specific setups.
        _setup_models(): Sets up the model.
        _setup_logger(): Sets up the logger.
        build_log_dict(p_dict, stage): Builds a dictionary for logging.
        _log_epoch(log_dict, update_cpkt): Logs information for an epoch.
        _update_best(log_dict): Updates the best performance metric.
        _save_checkpoint(tag): Saves a checkpoint of the model.
        load_checkpoint(tag): Loads a checkpoint of the model.
        train(): Trains the model.
        eval(): Evaluates the model.
    """

    def __init__(self, cfg):
        seed_everything(cfg.EXPERIMENT.SEED)
        self.cfg = cfg
        self._setup_logger()
        self.device = self._setup_device()
        self.current_epoch = 0
        self._setup_dataset()
        self._setup_models()
        self._setup_criterion()
        self._setup_optimizer()
        self._setup_scheduler()
        self._metric_specific_setups()
        self._basic_eval_setups()
        self._method_specific_setups()
        self._setup_resume()
        show_cfg(cfg, self.logger)

    def _setup_resume(self):
        current_checkpoint = os.path.join(
            self.log_path, f"current_{self.cfg.EXPERIMENT.SEED}"
        )

        if os.path.exists(current_checkpoint):
            print(f"Resuming model from {current_checkpoint}")
            self.load_checkpoint(f"current_{self.cfg.EXPERIMENT.SEED}")

    def _setup_device(self):
        return torch.device(
            self.cfg.EXPERIMENT.GPU if torch.cuda.is_available() else "cpu"
        )

    def _metric_specific_setups(self):
        self.metric_dict = metrics_dicts[self.cfg.METRIC]
        if self.metric_dict["best"] == "high":
            self.best_performance = float("-inf")
        elif self.metric_dict["best"] == "low":
            self.best_performance = float("inf")
        else:
            raise ValueError(
                f"Unsupported metric best type: {self.metric_dict['best']}"
            )

    def _setup_dataset(self):
        dataset = get_dataset(self.cfg)
        self.num_class = dataset["num_class"]
        self.biases = dataset["biases"]
        self.dataloaders = dataset["dataloaders"]
        self.sets = dataset["sets"]
        self.data_root = dataset["root"]
        self.target2name = dataset["target2name"]
        self.ba_groups = dataset["ba_groups"] if "ba_groups" in dataset else None

    def _setup_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.SOLVER.TYPE == "SGD":
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        elif self.cfg.SOLVER.TYPE == "Adam":
            self.optimizer = torch.optim.Adam(
                parameters,
                lr=self.cfg.SOLVER.LR,
                weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.cfg.SOLVER.TYPE}")

    def _setup_criterion(self):
        if self.cfg.SOLVER.CRITERION == "CE":
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported criterion type: {self.cfg.SOLVER.CRITERION}")

    def _loss_backward(self, loss):
        loss.backward()

    def _optimizer_step(self):
        self.optimizer.step()

    def _set_train(self):
        self.model.train()

    def _set_eval(self):
        self.model.eval()

    def _setup_scheduler(self):
        if self.cfg.SOLVER.SCHEDULER.TYPE == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.SOLVER.SCHEDULER.STEP_SIZE,
                gamma=self.cfg.SOLVER.SCHEDULER.LR_DECAY_RATE,
            )
        elif self.cfg.SOLVER.SCHEDULER.TYPE == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.cfg.SOLVER.SCHEDULER.LR_DECAY_STAGES,
                gamma=self.cfg.SOLVER.SCHEDULER.LR_DECAY_RATE,
            )
        elif self.cfg.SOLVER.SCHEDULER.TYPE == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.SOLVER.SCHEDULER.T_MAX
            )
        else:
            raise ValueError(
                f"Unsupported scheduler type: {self.cfg.SOLVER.SCHEDULER.TYPE}"
            )

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        loss = self.criterion(outputs, targets)
        self._loss_backward(loss)
        self._optimizer_step()
        return {"train_cls_loss": loss}

    def _train_epoch(self):
        self._set_train()
        self.current_lr = self.scheduler.get_last_lr()[0]
        avg_loss = None
        for batch in self.dataloaders["train"]:
            bsz = batch["targets"].shape[0]
            loss_dict = self._train_iter(batch)
            # initialize if needed
            if avg_loss is None:
                avg_loss = {key: AverageMeter() for key in loss_dict.keys()}
            # Update avg_loss for each key in loss_dict
            for key, value in loss_dict.items():
                avg_loss[key].update(value.item(), bsz)
        self.scheduler.step()
        avg_loss = {key: value.avg for key, value in avg_loss.items()}
        return avg_loss

    def _val_iter(self, batch):
        batch_dict = {}
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        loss = self.criterion(outputs, targets)
        batch_dict["predictions"] = torch.argmax(outputs, dim=1)
        batch_dict["targets"] = batch["targets"]
        for b in self.biases:
            batch_dict[b] = batch[b]
        return batch_dict, loss

    def _validate_epoch(self, stage="val"):
        self._set_eval()
        with torch.no_grad():
            all_data = {key: [] for key in self.biases}
            all_data["targets"] = []
            all_data["predictions"] = []

            losses = []
            for batch in self.dataloaders[stage]:
                batch_dict, loss = self._val_iter(batch)
                losses.append(loss.detach().cpu().numpy())
                for key, value in batch_dict.items():
                    all_data[key].append(value.detach().cpu().numpy())

            for key in all_data:
                all_data[key] = np.concatenate(all_data[key])
            # metric specific data
            if self.ba_groups is not None:
                all_data["ba_groups"] = self.ba_groups
            performance = get_performance[self.cfg.METRIC](all_data)
            performance["loss"] = np.mean(losses)
        return performance

    def _val_iter_tags(self, batch):
        batch_dict = {}

        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"]
        indices = batch["index"]

        tags_dict = {}  # zeros with shape of targets
        tags_dict = {
            f"{row['Class']}_{row['Tag']}": torch.zeros_like(targets)
            for _, row in self.overperforming_tags_df.iterrows()
        }

        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        batch_dict["predictions"] = torch.argmax(outputs, dim=1)
        batch_dict["targets"] = targets

        for i in range(len(indices)):
            class_name = self.target2name[targets[i].item()]
            irrelevant_tags = self.index_to_tags_test[indices[i].item()]
            irrelevant_tags = [tag.strip() for tag in irrelevant_tags.split(", ")]

            for tag in irrelevant_tags:
                if f"{class_name}_{tag}" in tags_dict:
                    tags_dict[f"{class_name}_{tag}"][i] = 1

        batch_dict.update(tags_dict)

        return batch_dict

    def _validate_epoch_tags(self, stage="val"):
        self._set_eval()
        with torch.no_grad():

            all_data = {
                f"{row['Class']}_{row['Tag']}": []
                for _, row in self.overperforming_tags_df.iterrows()
            }

            all_data["targets"] = []
            all_data["predictions"] = []

            for batch in self.dataloaders[stage]:
                batch_dict = self._val_iter_tags(batch)
                for key, value in batch_dict.items():
                    all_data[key].append(value.detach().cpu().numpy())

            for key in all_data:
                all_data[key] = np.concatenate(all_data[key])
            # metric specific data

            performance = get_performance[self.cfg.METRIC_TAGS](all_data)
        return performance

    def _method_specific_setups(self):
        pass

    def _basic_eval_setups(self):
        if self.cfg.METRIC == "wg_ovr_tags":
            self.tags_df_test = self.get_ram_tags(split="test")
            self.get_relevant_tags()
            self.tags_df_test = self.get_irrelevant_tags(
                self.tags_df_test, split="test"
            )
            self.index_to_tags_test = {
                row["index"]: (
                    row["irrelevant_tags"].replace(" | ", ", ")
                    if isinstance(row["irrelevant_tags"], str)
                    else " "
                )
                for _, row in self.tags_df_test.iterrows()
            }
            overperforming_tags_path = os.path.join(
                self.data_root, "overperforming_tags.csv"
            )
            if not os.path.exists(overperforming_tags_path):
                raise FileNotFoundError(
                    "File 'overperforming_tags.csv' not found. You should first train a vanilla model to calculate the overperforming tags."
                )
            # Load the DataFrame if the file exists
            print(f"Loading overperforming tags from {overperforming_tags_path}")
            self.overperforming_tags_df = pd.read_csv(overperforming_tags_path)
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
        batch_size = self.cfg.MITIGATOR.MAVIAS.LLM.BATCH_SIZE

        # Open the CSV file and read the tags
        unique_tags_df = pd.read_csv(
            os.path.join(self.data_root, "unique_tags_per_class.csv")
        )

        unique_tags_df["tags"] = unique_tags_df["tags"].apply(ast.literal_eval)

        self.relevant_tags = {}
        for target in unique_tags_df["class"]:
            path_to_check = os.path.join(
                self.data_root,
                "relevant_tags",
                f"{self.cfg.MITIGATOR.MAVIAS.LLM.TYPE}_bs{batch_size}_{target}_{self.target2name[target]}.csv",
            )
            # if path existis
            if not os.path.exists(path_to_check):
                raise FileNotFoundError(
                    "Relevant tags not found. You should first run MAVias to calculate the relevant tags."
                )
            else:
                print(f"Loading relevant tags from {path_to_check}")
                self.relevant_tags[target] = pd.read_csv(path_to_check)["tags"].tolist()

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

    def _setup_models(self):
        self.model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
            pretrained=self.cfg.MODEL.PRETRAINED,
        )
        if self.cfg.MODEL.FREEZE_BACKBONE:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

        self.model.to(self.device)

    def _setup_logger(self):
        tags = self.cfg.EXPERIMENT.TAG.split(",")
        experiment_name = os.path.join(
            self.cfg.EXPERIMENT.PROJECT,
            self.cfg.EXPERIMENT.TAG,
            self.cfg.EXPERIMENT.NAME,
        )
        self.log_path = os.path.join(self.cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.logger = setup_logger(os.path.join(self.log_path, "out.log"))
        if self.cfg.LOG.WANDB:
            try:
                import wandb

                wandb.init(
                    project=self.cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags
                )
            except ImportError as e:
                print(log_msg(f"Failed to import WANDB: {e}", "INFO", self.logger))
                self.cfg.LOG.WANDB = False
            except Exception as e:
                print(log_msg(f"Failed to use WANDB: {e}", "INFO", self.logger))
                self.cfg.LOG.WANDB = False

        self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

    def build_log_dict(self, p_dict, stage="train"):
        log_dict = {}
        for key, value in p_dict.items():
            log_dict[f"{stage}_{key}"] = value
        return log_dict

    def _log_epoch(self, log_dict, update_cpkt):
        # tensorboard log
        for k, v in log_dict.items():
            if isinstance(v, (int, float)):  # Log only numerical values
                self.tf_writer.add_scalar(k, v, self.current_epoch)
            # else:
            #     print(f"Skipping {k}: Value is not a number ({v})")  # Optional: Debugging message
        self.tf_writer.flush()
        # wandb log
        if self.cfg.LOG.WANDB:
            import wandb

            wandb.log({"current lr": self.current_lr})
            wandb.log(log_dict)
        if update_cpkt:
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_performance"] = self.best_performance
        # worklog.txt
        # Write to out.log with keys as columns

        log_file_path = os.path.join(self.log_path, "out.log")
        log_keys = ["epoch", "lr"] + list(log_dict.keys())
        column_width = (
            max(len(key) for key in log_keys) + 2
        )  # Adjust column width dynamically

        if self.current_epoch == 0:  # Create headers if file is new
            with open(log_file_path, "a", encoding="utf-8") as writer:
                # Header
                header = "".join(f"{key:<{column_width}}" for key in log_keys)
                writer.write(header + os.linesep)
                writer.write("-" * len(header) + os.linesep)

        with open(log_file_path, "a", encoding="utf-8") as writer:
            # Row data
            row = f"{self.current_epoch:<{column_width}}{self.current_lr:<{column_width}.6f}"
            # row += "".join(
            #     f"{log_dict[key]:<{column_width}.4f}" for key in log_dict.keys()
            # )
            for key in log_dict.keys():
                value = log_dict[key]
                if isinstance(value, (int, float)):  # Format numbers
                    row += f"{value:<{column_width}.4f}"
                else:  # Format strings and other non-numeric values
                    row += f"{str(value):<{column_width}}"
            writer.write(row + os.linesep)

        # Optional: Write to CSV for visualization
        csv_file_path = os.path.join(self.log_path, "logs.csv")
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, "w", encoding="utf-8") as csv_file:
                csv_file.write(",".join(log_keys) + os.linesep)

        with open(csv_file_path, "a", encoding="utf-8") as csv_file:
            row = [self.current_epoch, self.current_lr] + list(log_dict.values())
            csv_file.write(",".join(map(str, row)) + os.linesep)

    def _update_best(self, log_dict):
        if (
            log_dict[f"{self.cfg.LOG.SAVE_CRITERION}_{self.metric_dict['performance']}"]
            > self.best_performance
            and self.metric_dict["best"] == "high"
        ):
            self.best_performance = log_dict[
                f"{self.cfg.LOG.SAVE_CRITERION}_{self.metric_dict['performance']}"
            ]
            return True
        if (
            log_dict[f"{self.cfg.LOG.SAVE_CRITERION}_{self.metric_dict['performance']}"]
            < self.best_performance
            and self.metric_dict["best"] == "low"
        ):
            self.best_performance = log_dict[
                f"{self.cfg.LOG.SAVE_CRITERION}_{self.metric_dict['performance']}"
            ]
            return True
        return False

    def _save_checkpoint(self, tag):
        state = {
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_performance": self.best_performance,
            "scheduler": self.scheduler.state_dict(),
        }
        save_checkpoint(state, os.path.join(self.log_path, tag))

    def load_checkpoint(self, tag):
        checkpoint = load_checkpoint(os.path.join(self.log_path, tag))
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.best_performance = checkpoint["best_performance"]
        self.current_epoch = checkpoint["epoch"]
        print(f"Loaded checkpoint from {os.path.join(self.log_path, tag)}")
        print(
            log_msg(
                f"Loaded checkpoint from {os.path.join(self.log_path, tag)}",
                "INFO",
                self.logger,
            )
        )

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
                if self.cfg.METRIC == "wg_ovr_tags":
                    raise ValueError(
                        f"the self.cfg.LOG.TRAIN_PERFORMANCE should be False for wg_ovr_tags metric, you gave {self.cfg.LOG.TRAIN_PERFORMANCE}"
                    )
                train_performance = self._validate_epoch(stage="train")
                train_log_dict = self.build_log_dict(train_performance, stage="train")
                log_dict.update(train_log_dict)
            if self.cfg.LOG.SAVE_CRITERION == "val":
                if self.cfg.METRIC == "wg_ovr_tags":
                    raise ValueError(
                        f"the self.cfg.LOG.SAVE_CRITERION should be test for wg_ovr_tags metric, you gave {self.cfg.LOG.SAVE_CRITERION}"
                    )
                val_performance = self._validate_epoch(stage="val")
                val_log_dict = self.build_log_dict(val_performance, stage="val")
                log_dict.update(val_log_dict)
            if self.cfg.METRIC == "wg_ovr_tags":
                test_performance = self._validate_epoch_tags(stage="test")
            else:
                test_performance = self._validate_epoch(stage="test")
            test_log_dict = self.build_log_dict(test_performance, stage="test")
            log_dict.update(test_log_dict)
            update_cpkt = self._update_best(log_dict)
            if update_cpkt:
                self._save_checkpoint(tag=f"best_{self.cfg.EXPERIMENT.SEED}")
            self._save_checkpoint(tag=f"current_{self.cfg.EXPERIMENT.SEED}")
            self._log_epoch(log_dict, update_cpkt)
        self._save_checkpoint(tag="latest")

    def eval(self):
        self.load_checkpoint(self.cfg.MODEL.PATH)
        test_performance = self._validate_epoch(stage="test")
        test_log_dict = self.build_log_dict(test_performance, stage="test")
        print(log_msg(f"Test performance: {test_log_dict}", "EVAL", self.logger))
