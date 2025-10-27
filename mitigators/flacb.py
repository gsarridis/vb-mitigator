import os
from models.utils import get_local_model_dict
import torch
import numpy as np
from .base_trainer import BaseTrainer
from models.builder import get_model
import torch.nn.functional as F


def pairwise_distances(a, b=None, eps=1e-6):
    """
    Calculates the pairwise distances between matrices a and b (or a and a, if b is not set)
    :param a:
    :param b:
    :return:
    """
    if b is None:
        b = a

    aa = torch.sum(a**2, dim=1)
    bb = torch.sum(b**2, dim=1)

    aa = aa.expand(bb.size(0), aa.size(0)).t()
    bb = bb.expand(aa.size(0), bb.size(0))

    AB = torch.mm(a, b.transpose(0, 1))

    dists = aa + bb - 2 * AB
    dists = torch.clamp(dists, min=0, max=np.inf)
    dists = torch.sqrt(dists + eps)
    return dists


def flac_loss(protected_attr_features, features, labels, d=1):
    protected_attr_features = F.normalize(protected_attr_features, dim=1)
    features = F.normalize(features, dim=1)
    # Protected attribute features kernel
    protected_d = pairwise_distances(protected_attr_features)
    protected_s = 1.0 / (1 + protected_d**d)
    # Target features kernel
    features_d = pairwise_distances(features)
    features_s = 1.0 / (1 + features_d**d)

    th = (torch.max(protected_s) + torch.min(protected_s)) / 2
    # calc the mask
    mask = (labels[:, None] == labels) & (protected_s < th) | (
        labels[:, None] != labels
    ) & (protected_s > th)
    mask = mask.to(labels.device)

    # if mask is empty, return zero
    if sum(sum(mask)) == 0:
        return torch.tensor(0.0).to(labels.device)
    # similarity to distance
    protected_s = 1 - protected_s

    # convert to probabilities
    protected_s = protected_s / (
        torch.sum(protected_s * mask.int().float(), dim=1, keepdim=True) + 1e-7
    )
    features_s = features_s / (
        torch.sum(features_s * mask.int().float(), dim=1, keepdim=True) + 1e-7
    )

    # Jeffrey's divergence
    loss = (protected_s[mask] - features_s[mask]) * (
        torch.log(protected_s[mask]) - torch.log(features_s[mask])
    )

    return torch.mean(loss)


class FLACBTrainer(BaseTrainer):

    def _setup_models(self):
        self.model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
            pretrained=self.cfg.MODEL.PRETRAINED,
        )
        self.model.to(self.device)
        
        bcc_net_dict = get_local_model_dict(self.cfg.MITIGATOR.FLACB.BCC_PATH)
        self.bcc_net = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
        )
        self.bcc_net.load_state_dict(bcc_net_dict["model"])

        self.bcc_net.to(self.device)
        self.bcc_net.eval()

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        loss_flac = 0
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if not isinstance(outputs, tuple):
            raise ValueError("Model output must be a tuple (logits, features)")
        outputs, features = outputs

        with torch.no_grad():
            _, pr_feat = self.bcc_net(inputs)
        loss_flac += self.cfg.MITIGATOR.FLACB.LOSS.ALPHA * flac_loss(
            pr_feat, features, targets, self.cfg.MITIGATOR.FLACB.LOSS.DELTA
        )
        loss_cl = self.criterion(outputs, targets)
        loss = self.cfg.MITIGATOR.FLACB.LOSS.CE_WEIGHT * loss_cl + loss_flac
        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()
        return {"train_cls_loss": loss_cl, "train_flac_loss": loss_flac}
