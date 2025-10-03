from .base_trainer import BaseTrainer
import ast
import os

import pandas as pd

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from my_datasets.builder import get_dataset
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
import mitigators.losses as losses
import torch.nn as nn
import torch.nn.functional as F


def disentanglement_loss(features):
    loss = 0
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            f_i = features[i] - features[i].mean(dim=0, keepdim=True)
            f_j = features[j] - features[j].mean(dim=0, keepdim=True)
            cov = (f_i.T @ f_j) / f_i.size(0)  # covariance matrix
            loss += (cov**2).sum()  # penalize off-diagonal
    return loss / (n * (n - 1) / 2)


def estimate_mutual_info(x, y, temperature=0.1):
    """
    Estimate mutual information between two feature slices x and y using InfoNCE.

    Args:
        x: Tensor of shape [batch_size, dim_x]
        y: Tensor of shape [batch_size, dim_y]
        temperature: scaling factor for logits
    Returns:
        Scalar tensor: estimated negative mutual information (to minimize)
    """
    batch_size = x.size(0)

    # Normalize features (optional but stabilizes)
    x_norm = F.normalize(x, dim=1)
    y_norm = F.normalize(y, dim=1)

    # Compute similarity matrix [batch_size, batch_size]
    sim_matrix = x_norm @ y_norm.T / temperature  # cosine similarity scaled

    # InfoNCE target: diagonal elements are positive pairs
    labels = torch.arange(batch_size, device=x.device)

    # Cross-entropy loss over rows
    mi_loss = F.cross_entropy(sim_matrix, labels)
    return mi_loss


def orthogonal_gradient_penalty(logits, slices, targets):
    """
    Enforces orthogonal gradients between feature slices.
    """
    penalty = 0.0
    num_pairs = 0
    n = len(slices)

    # 1. Compute the overall cross-entropy loss for the combined logits
    # The gradients will be computed w.r.t this single loss
    loss_ce = torch.nn.functional.cross_entropy(logits, targets)

    # 2. Compute the gradient of the overall loss with respect to EACH slice
    grads = []
    for i in range(n):
        # We need to ensure each slice is a leaf variable requiring gradients
        if not slices[i].requires_grad:
            slices[i].requires_grad_(True)

        # We need retain_graph=True because we'll be doing multiple backward passes
        # with respect to the same graph (the forward pass that produced logits)
        grad_i = torch.autograd.grad(
            outputs=loss_ce, inputs=slices[i], create_graph=True, retain_graph=True
        )[0]
        grads.append(grad_i)

    # 3. Calculate the pairwise cosine similarity of the gradients
    for i in range(n):
        for j in range(i + 1, n):
            grad_i = grads[i].view(grad_i.size(0), -1)  # Flatten for cosine_similarity
            grad_j = grads[j].view(grad_j.size(0), -1)

            # Cosine similarity for each example in the batch, then average
            cos_sim = torch.nn.functional.cosine_similarity(grad_i, grad_j, dim=1)
            penalty += cos_sim.mean()
            num_pairs += 1

    # Average the penalty over all unique pairs of slices
    if num_pairs > 0:
        penalty /= num_pairs

    return penalty, loss_ce


class ERMDevTrainer(BaseTrainer):

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, features = outputs

        # loss_ce = self.criterion(outputs[0], targets)
        # loss_ce += self.criterion(outputs[1], targets)
        logits = outputs[0] + outputs[1]
        mi_loss, loss_ce = orthogonal_gradient_penalty(logits, features, targets)

        loss = mi_loss + loss_ce
        self._loss_backward(loss)
        self._optimizer_step()
        return {"train_cls_loss": loss_ce, "train_dis_loss": mi_loss}

    def _val_iter(self, batch, part):
        batch_dict = {}
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        outputs = self.model(inputs, part=part)
        if isinstance(outputs, tuple):
            outputs, _ = outputs

        if isinstance(outputs, list):
            loss = self.criterion(outputs[0], targets)
            loss += self.criterion(outputs[1], targets)
        else:
            loss = self.criterion(outputs, targets)
        batch_dict["predictions"] = torch.argmax(outputs, dim=1)
        batch_dict["targets"] = batch["targets"]
        for b in self.biases:
            batch_dict[b] = batch[b]
        return batch_dict, loss

    def _validate_epoch(self, stage="val"):
        self._set_eval()
        with torch.no_grad():
            for part in range(1, 3):
                all_data = {key: [] for key in self.biases}
                all_data["targets"] = []
                all_data["predictions"] = []

                losses = []
                show_progress_bar = self.cfg.EXPERIMENT.PROGRESS_BAR

                # Set up the dataloader iterator
                dataloader_iterator = self.dataloaders[stage]

                # Conditionally wrap the dataloader with tqdm for a progress bar
                if show_progress_bar:
                    # Create a tqdm progress bar
                    progress_bar = tqdm(
                        dataloader_iterator,
                        # Assumes self.current_epoch and self.epochs are available for a more descriptive bar
                        desc=f"Epoch {getattr(self, 'current_epoch', '?')}/{self.cfg.SOLVER.EPOCHS} Eval on {stage} set",
                        unit="batch",
                    )
                else:
                    # If no progress bar, just use the original dataloader
                    progress_bar = dataloader_iterator

                for batch in progress_bar:
                    batch_dict, loss = self._val_iter(batch, part)
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
                print(performance)

        return performance
