import torch
import numpy as np

from models.utils import get_local_model_dict
from .base_trainer import BaseTrainer
from models.builder import get_model, get_bcc, get_local_bccs
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


class FLACTrainer(BaseTrainer):

    def _setup_models(self):
        super()._setup_models()

        bcc_paths = list(self.cfg.MITIGATOR.FLAC.BCC_PATHS)
        if not bcc_paths and self.cfg.MITIGATOR.FLAC.BCC_PATH != "":
            bcc_paths = [self.cfg.MITIGATOR.FLAC.BCC_PATH]

        if bcc_paths:
            self.bcc_nets = get_local_bccs(
                self.cfg, bcc_paths, self.num_class, self.device, self.biases
            )
        else:
            self.bcc_nets = get_bcc(self.cfg, self.num_class)

            for _, bcc_net in self.bcc_nets.items():
                bcc_net.to(self.device)
                bcc_net.eval()

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        loss_flac = 0

        outputs = self.model(inputs)
        if not isinstance(outputs, tuple):
            raise ValueError("Model output must be a tuple (logits, features)")
        outputs, features = outputs

        for _, bcc_net in self.bcc_nets.items():
            with torch.no_grad():
                _, pr_feat = bcc_net(inputs)
            loss_flac += self.cfg.MITIGATOR.FLAC.LOSS.ALPHA * flac_loss(
                pr_feat, features, targets, self.cfg.MITIGATOR.FLAC.LOSS.DELTA
            )
        loss_cl = self.cfg.MITIGATOR.FLAC.LOSS.CE_WEIGHT * self.criterion(
            outputs, targets
        )
        loss = loss_cl + loss_flac
        self.optimizer.zero_grad()
        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()
        return {"train_cls_loss": loss_cl, "train_flac_loss": loss_flac}
