import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from .losses import BBLoss
from .base_trainer import BaseTrainer


class BBTrainer(BaseTrainer):

    def _setup_criterion(self):
        super()._setup_criterion()
        # Example inputs
        all_targets = []
        all_biases = []
        all_biases2 = []

        if len(self.biases) > 2:
            raise ValueError("not implemented")

        for data in self.dataloaders["train"]:

            # Collect targets
            all_targets.append(data["targets"])
            all_biases.append(data[self.biases[0]])
            if len(self.biases)>1:
                all_biases2.append(data[self.biases[1]])

        # Concatenate all collected data
        all_targets = torch.cat(all_targets, dim=0)
        all_biases = torch.cat(all_biases, dim=0)
        if len(self.biases)>1:
            all_biases2 = torch.cat(all_biases2, dim=0)

        # Step 2: Compute the confusion matrix
        conf_matrix = confusion_matrix(all_targets, all_biases)
        if len(self.biases)>1:
            conf_matrix2 = confusion_matrix(all_targets, all_biases2)
        # print(conf_matrix)
        self.criterion_train = BBLoss(torch.tensor(conf_matrix))
        if len(self.biases)>1:
            self.criterion_train2 = BBLoss(torch.tensor(conf_matrix2))

    def _train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        biases = batch[self.biases[0]].to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs
        loss = self.criterion_train(outputs, targets, biases)
        if len(self.biases)>1:
            loss += self.criterion_train2(outputs, targets, biases)
        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()
        return {"train_cls_loss": loss}
