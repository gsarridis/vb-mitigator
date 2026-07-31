"""
SAE-steering mitigator (post-hoc debiasing).

Pipeline:
  1. Load a configurable, already-trained classifier (e.g. ERM baseline), freeze it.
  2. Train a supervised SAE on that frozen model's penultimate features over a BALANCED
     set. For each bias attribute ``b`` (in ``self.biases``) a dedicated group of ``K_b``
     SAE latent neurons is supervised (cross-entropy, one neuron per class) so those
     neurons become the spurious-attribute detectors.
  3. Steer: zero the dedicated neurons in the SAE reconstruction (all attributes, or the
     subset in ``MITIGATOR.SAE.STEER_ATTRS``).
  4. Retrain a fresh last layer (DFR) on the steered features over the balanced set.
  5. Evaluate original vs steered(old head) vs steered+retrained head.

The SAE is adapted from the AutoEncoder in sae-for-vlm/dictionary_learning/dictionary.py
(vendored here to avoid that package's einops/wandb deps).
"""

import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mitigators.base_trainer import BaseTrainer
from models.builder import get_model
from tools.metrics.sae_steer import sae_steer
from tools.metrics.utils import AverageMeter
from tools.utils import log_msg, save_checkpoint, load_checkpoint


class SupervisedSAE(nn.Module):
    """One-layer SAE with the first ``sum(group_sizes)`` latents supervised in groups."""

    def __init__(self, activation_dim, dict_size, group_sizes):
        super().__init__()
        assert dict_size >= sum(group_sizes), "dict_size must exceed #supervised neurons"
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(torch.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)

        # init (same scheme as AutoEncoder)
        w = torch.randn(activation_dim, dict_size)
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

        self.group_sizes = list(group_sizes)
        self.n_sup = sum(self.group_sizes)
        self.group_slices = []
        start = 0
        for k in self.group_sizes:
            self.group_slices.append((start, start + k))
            start += k

    def encode(self, x):
        return torch.relu(self.encoder(x - self.bias))

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

    def sup_logits(self, f, group_id):
        s, e = self.group_slices[group_id]
        return f[:, s:e]

    def ablate(self, f, group_ids, value=0.0):
        f = f.clone()
        for gi in group_ids:
            s, e = self.group_slices[gi]
            f[:, s:e] = value
        return f

    def steer(self, x, group_ids, value=0.0):
        return self.decode(self.ablate(self.encode(x), group_ids, value))


class SAESteerTrainer(BaseTrainer):

    # ----- setup ----------------------------------------------------------- #
    def _setup_models(self):
        cfg = self.cfg
        self.model = get_model(cfg.MODEL.TYPE, self.num_class, pretrained=cfg.MODEL.PRETRAINED)
        path = cfg.MITIGATOR.SAE.MODEL_PATH
        if not path:
            raise ValueError("MITIGATOR.SAE.MODEL_PATH must point to a trained classifier checkpoint.")
        ckpt = load_checkpoint(path)
        self.model.load_state_dict(ckpt["model"])
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device).eval()
        print(log_msg(f"Loaded frozen classifier from {path}", "INFO", self.logger))

        embed = getattr(self.model, "embed_size", None)
        if embed is None:
            raise ValueError("Steered model must expose `.embed_size` and return (logits, feat).")

        group_sizes = self._infer_group_sizes()  # K_b per bias attribute
        dict_size = cfg.MITIGATOR.SAE.EXPANSION_FACTOR * embed
        self.sae = SupervisedSAE(embed, dict_size, group_sizes).to(self.device)
        self.steer_head = nn.Linear(embed, self.num_class).to(self.device)

        # which attribute-groups to ablate when steering ([] -> all)
        steer_attrs = list(cfg.MITIGATOR.SAE.STEER_ATTRS) or list(self.biases)
        self.steer_group_ids = [self.biases.index(b) for b in steer_attrs]
        self.steer_value = cfg.MITIGATOR.SAE.STEER_VALUE
        print(log_msg(
            f"SAE dict_size={dict_size}, supervised groups={dict(zip(self.biases, group_sizes))}, "
            f"steering groups={self.steer_group_ids}", "INFO", self.logger))

    def _infer_group_sizes(self):
        """K_b = number of classes of each bias attribute, from the train set."""
        train_set = self.sets.get("train")
        sizes = []
        for b in self.biases:
            if train_set is not None and hasattr(train_set, "bias") and b in getattr(train_set, "bias", {}):
                sizes.append(int(np.max(train_set.bias[b]) + 1))
            else:  # fallback: scan a few train batches
                mx = 0
                for i, batch in enumerate(self.dataloaders["train"]):
                    mx = max(mx, int(batch[b].max().item()))
                    if i >= 50:
                        break
                sizes.append(mx + 1)
        return sizes

    def _setup_optimizer(self):
        cfg = self.cfg
        self.sae_optimizer = torch.optim.Adam(
            self.sae.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
        self.optimizer = self.sae_optimizer  # so base _setup_scheduler builds on it

    def _set_train(self):
        self.model.eval()  # frozen classifier always in eval

    def _set_eval(self):
        self.model.eval()
        self.sae.eval()
        self.steer_head.eval()

    # ----- phase A: SAE training ------------------------------------------- #
    def _sae_train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        with torch.no_grad():
            _, feat = self.model(inputs)

        self.sae_optimizer.zero_grad()
        x_hat, f = self.sae(feat)
        recon = (feat - x_hat).pow(2).sum(dim=-1).mean()
        l1 = f[:, self.sae.n_sup:].norm(p=1, dim=-1).mean()  # exclude dedicated neurons
        sup = 0.0
        for gi, b in enumerate(self.biases):
            sup = sup + self.criterion(self.sae.sup_logits(f, gi), batch[b].to(self.device))

        loss = recon + self.cfg.MITIGATOR.SAE.L1_PENALTY * l1 + self.cfg.MITIGATOR.SAE.SUP_PENALTY * sup
        loss.backward()
        self.sae_optimizer.step()
        if getattr(self, "scheduler", None) is not None:
            self.scheduler.step()
        return {"loss": loss, "recon": recon, "l1": l1, "sup": sup}

    def _head_train_iter(self, batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)
        with torch.no_grad():
            _, feat = self.model(inputs)
            steered = self.sae.steer(feat, self.steer_group_ids, self.steer_value)

        self.head_optimizer.zero_grad()
        loss = self.criterion(self.steer_head(steered), targets)
        loss.backward()
        self.head_optimizer.step()
        return {"loss": loss}

    def _run_epoch(self, iter_fn, desc):
        bar = self.dataloaders["train"]
        if self.cfg.EXPERIMENT.PROGRESS_BAR:
            bar = tqdm(bar, desc=desc, unit="batch")
        meters = {}
        for batch in bar:
            losses = iter_fn(batch)
            bsz = batch["targets"].shape[0]
            for k, v in losses.items():
                meters.setdefault(k, AverageMeter()).update(float(v), bsz)
        return {k: m.avg for k, m in meters.items()}

    # ----- evaluation (3 conditions) --------------------------------------- #
    def _validate_epoch(self, stage="val"):
        self._set_eval()
        data = {"targets": [], "orig": [], "steered_orig": [], "steered_dfr": []}
        for b in self.biases:
            data[b] = []
            data[f"sup_pred_{b}"] = []

        with torch.no_grad():
            for batch in self.dataloaders[stage]:
                inputs = batch["inputs"].to(self.device)
                logits, feat = self.model(inputs)
                f = self.sae.encode(feat)
                f_ablated = self.sae.ablate(f, self.steer_group_ids, self.steer_value)
                steered = self.sae.decode(f_ablated)

                data["targets"].append(batch["targets"].numpy())
                data["orig"].append(logits.argmax(1).cpu().numpy())
                data["steered_orig"].append(self.model.fc(steered).argmax(1).cpu().numpy())
                data["steered_dfr"].append(self.steer_head(steered).argmax(1).cpu().numpy())
                for gi, b in enumerate(self.biases):
                    data[b].append(batch[b].numpy())
                    data[f"sup_pred_{b}"].append(self.sae.sup_logits(f, gi).argmax(1).cpu().numpy())

        for k in list(data):
            data[k] = np.concatenate(data[k])
        data["ba_groups"] = self.ba_groups if self.ba_groups is not None else []
        return sae_steer(data)

    # ----- two-phase training ---------------------------------------------- #
    def train(self):
        # Phase A: train the supervised SAE; select by mean dedicated-neuron accuracy.
        best_sup = -float("inf")
        for epoch in range(1, self.cfg.SOLVER.EPOCHS + 1):
            self.current_epoch = epoch
            self.current_lr = self.optimizer.param_groups[0]["lr"]
            self.sae.train()
            losses = self._run_epoch(self._sae_train_iter, f"[SAE] epoch {epoch}")
            perf = self._validate_epoch(stage="val")
            sup = np.mean([perf[f"sup_acc_{b}"] for b in self.biases])
            print(log_msg(
                f"[SAE] epoch {epoch} losses={ {k: round(v,4) for k,v in losses.items()} } "
                f"val sup_acc={sup:.4f} steered_orig_wg={perf['steered_orig_wg']:.4f}",
                "TRAIN", self.logger))
            if sup > best_sup:
                best_sup = sup
                self._save_checkpoint("best_sae")
        self._load_sae("best_sae")

        # Phase B: DFR last-layer retraining on the steered features.
        self.head_optimizer = torch.optim.Adam(self.steer_head.parameters(), lr=self.cfg.MITIGATOR.SAE.HEAD_LR)
        for epoch in range(1, self.cfg.MITIGATOR.SAE.HEAD_EPOCHS + 1):
            self.current_epoch = epoch
            self.steer_head.train()
            losses = self._run_epoch(self._head_train_iter, f"[HEAD] epoch {epoch}")
            perf = self._validate_epoch(stage="val")
            print(log_msg(
                f"[HEAD] epoch {epoch} loss={losses['loss']:.4f} "
                f"val steered_dfr_wg={perf['steered_dfr_wg']:.4f} steered_dfr_bc={perf['steered_dfr_bc']:.4f}",
                "TRAIN", self.logger))
            if perf["steered_dfr_wg"] >= self.best_performance:
                self.best_performance = perf["steered_dfr_wg"]
                self._save_checkpoint("best")
        self.load_checkpoint("best")

        test_perf = self._validate_epoch(stage="test")
        print(log_msg(f"[TEST] {self._fmt(test_perf)}", "EVAL", self.logger))

    def eval(self):
        self.load_checkpoint("best")
        test_perf = self._validate_epoch(stage="test")
        print(log_msg(f"[TEST] {self._fmt(test_perf)}", "EVAL", self.logger))

    @staticmethod
    def _fmt(perf):
        return {k: round(v, 4) for k, v in perf.items() if isinstance(v, (int, float))}

    # ----- checkpoints ----------------------------------------------------- #
    def _save_checkpoint(self, tag):
        state = {
            "epoch": self.current_epoch,
            "sae": self.sae.state_dict(),
            "steer_head": self.steer_head.state_dict(),
            "group_sizes": self.sae.group_sizes,
            "best_performance": self.best_performance,
        }
        save_checkpoint(state, os.path.join(self.log_path, tag))

    def _load_sae(self, tag):
        ckpt = load_checkpoint(os.path.join(self.log_path, tag))
        self.sae.load_state_dict(ckpt["sae"])

    def load_checkpoint(self, tag):
        ckpt = load_checkpoint(os.path.join(self.log_path, tag))
        self.sae.load_state_dict(ckpt["sae"])
        self.steer_head.load_state_dict(ckpt["steer_head"])
        self.best_performance = ckpt.get("best_performance", self.best_performance)
