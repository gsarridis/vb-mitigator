"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# --------------------------------------------------------
# implementation from Gradient Starvation:
# https://github.com/mohammadpz/Gradient_Starvation
# --------------------------------------------------------

import torch


from tqdm import tqdm
from .base_trainer import BaseTrainer


class SpectralDecoupleTrainer(BaseTrainer):

    def _setup_optimizer(self):
        if self.cfg.SOLVER.TYPE == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.SOLVER.LR,
                momentum=self.cfg.SOLVER.MOMENTUM,
                weight_decay=0.0,
            )
        elif self.cfg.SOLVER.TYPE == "Adam"  or self.cfg.SOLVER.TYPE == "AdamW":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.SOLVER.LR,
                weight_decay=0.0,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.cfg.SOLVER.TYPE}")



    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        ce_loss = self.criterion(outputs, targets)
        logits_norm = ((outputs[range(outputs.shape[0]), targets]) ** 2).mean()
        loss = ce_loss + self.cfg.MITIGATOR.SD.COEF * logits_norm
        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()
        return {"train_cls_loss": ce_loss, "train_norm": logits_norm}
    

