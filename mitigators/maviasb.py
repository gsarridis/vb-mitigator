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

from models.builder import get_bcc, get_model
from models.simple_mlp import SimpleMLP
from models.utils import get_local_model_dict
from .base_trainer import BaseTrainer
from tools.utils import load_ollama_docker
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage


class MAVIASBTrainer(BaseTrainer):

    def _setup_models(self):
        self.model = get_model(
            self.cfg.MODEL.TYPE, self.num_class, pretrained=self.cfg.MODEL.PRETRAINED
        )

        self.model.to(self.device)

        if self.cfg.MITIGATOR.MAVIASB.BCC_PATH != "":
            bcc_net_dict = get_local_model_dict(self.cfg.MITIGATOR.MAVIASB.BCC_PATH)
            self.bcc_net = get_model(
                self.cfg.MODEL.TYPE,
                self.num_class,
            )
            self.bcc_net.load_state_dict(bcc_net_dict["model"])

            self.bcc_net.to(self.device)
            self.bcc_net.eval()
            self.bcc_nets = {self.biases[0]: self.bcc_net}
        else:
            self.bcc_nets = get_bcc(self.cfg, self.num_class)

            for _, bcc_net in self.bcc_nets.items():
                bcc_net.to(self.device)
                bcc_net.eval()

        try:
            in_features = self.model.fc.in_features
        except:
            in_features = 512
        self.proj_net = SimpleMLP(in_features, in_features)
        self.proj_net.to(self.device)

    def _setup_optimizer(self):
        super()._setup_optimizer()
        parameters_projection = self.proj_net.parameters()

        if self.cfg.MITIGATOR.MAVIASB.PROJNET.OPTIM.TYPE == "SGD":
            self.optimizer_projection = torch.optim.SGD(
                parameters_projection,
                lr=self.cfg.MITIGATOR.MAVIASB.PROJNET.OPTIM.LR,
                momentum=self.cfg.MITIGATOR.MAVIASB.PROJNET.OPTIM.MOMENTUM,
                weight_decay=self.cfg.MITIGATOR.MAVIASB.PROJNET.OPTIM.WEIGHT_DECAY,
            )
        elif self.cfg.MITIGATOR.MAVIASB.PROJNET.OPTIM.TYPE == "Adam":
            self.optimizer_projection = torch.optim.Adam(
                parameters_projection,
                lr=self.cfg.MITIGATOR.MAVIASB.PROJNET.OPTIM.LR,
                weight_decay=self.cfg.MITIGATOR.MAVIASB.PROJNET.OPTIM.WEIGHT_DECAY,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer type: {self.cfg.MITIGATOR.MAVIASB.PROJNET.OPTIM.TYPE}"
            )

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        indices = batch["index"]

        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_projection.zero_grad(set_to_none=True)
        with torch.no_grad():
            for _, bcc_net in self.bcc_nets.items():
                _, b_feats = bcc_net(inputs)
        b_feats = self.proj_net(b_feats)

        logits, logits2 = self.model.mavias_forward(inputs, b_feats)
        tmp = logits2.detach().cpu().clone()
        norm_main = torch.norm(logits)
        norm_clip = torch.norm(tmp).to(self.device)
        norm_loss = F.mse_loss(
            norm_main, norm_clip * self.cfg.MITIGATOR.MAVIASB.LOSS.LAMBDA
        )
        ce_loss = self.criterion(logits + logits2, targets)

        loss = ce_loss + self.cfg.MITIGATOR.MAVIASB.LOSS.ALPHA * norm_loss

        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()

        return {"train_cls_loss": ce_loss, "train_norm_loss": norm_loss}

    def _optimizer_step(self):
        self.optimizer.step()
        self.optimizer_projection.step()

    def _set_train(self):
        self.proj_net.train()
        for _, bcc_net in self.bcc_nets.items():
            bcc_net.eval()
        return super()._set_train()

    def _set_eval(self):
        self.proj_net.eval()
        for _, bcc_net in self.bcc_nets.items():
            bcc_net.eval()
        return super()._set_eval()
