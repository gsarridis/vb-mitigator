import torch
from .base_trainer import BaseTrainer


class ERMBCCTrainer(BaseTrainer):
    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch[self.biases[0]].to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        loss = self.criterion(outputs, targets)
        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()
        return {"train_cls_loss": loss}

    def _val_iter(self, batch):
        batch_dict = {}
        inputs = batch["inputs"].to(self.device)
        targets = batch[self.biases[0]].to(self.device)
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        loss = self.criterion(outputs, targets)
        batch_dict["predictions"] = torch.argmax(outputs, dim=1)
        batch_dict["targets"] = batch[self.biases[0]]
        for b in self.biases:
            batch_dict[b] = batch[b]
        return batch_dict, loss
