from .base_trainer import BaseTrainer
from tools.metrics.utils import AverageMeter
import torch
from torchvision import transforms as T
from my_datasets.utils import TwoCropTransform
from my_datasets.celeba import get_celeba
from my_datasets.cifar100 import get_cifar100_loaders
from my_datasets.imagenet9 import get_imagenet9l
from my_datasets.urbancars import get_urbancars_loader
from my_datasets.biased_mnist import get_color_mnist
from my_datasets.fb_biased_mnist import get_color_mnist as get_fb_color_mnist
from my_datasets.utk_face import get_utk_face
from my_datasets.waterbirds import get_waterbirds
from my_datasets.cifar10 import get_cifar10_loaders
from my_datasets.stanford_dogs import get_stanford_dogs_loader
from models.utils import get_local_model_dict
from models.builder import get_model
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity


class DebiasSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, biases=None, mask=None):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("Num of labels does not match num of features")

        if mask is None:
            assert biases is not None
            biases = biases.contiguous().view(-1, 1)
            label_mask = torch.eq(labels, labels.T)
            bias_mask = torch.ne(biases, biases.T)
            mask = label_mask & bias_mask
            mask = mask.float().cuda()
        else:
            label_mask = torch.eq(labels, labels.T).float().cuda()
            mask = label_mask * mask

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).cuda(),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # compute mean of log-likelihood over positive
        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1
        mean_log_prob_pos = (mask * log_prob).sum(1) / sum_mask

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss


class UnsupBiasContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.con_loss = DebiasSupConLoss(temperature=temperature)
        print(f"UnsupBiasContrastiveLoss - T: {self.temperature}")

    def cosine_pairwise(self, x):
        x = x.permute((1, 0))
        cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-1)
        cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
        return cos_sim_pairwise

    def forward(self, logits, labels, cont_features, cont_labels, cont_bias_feats):
        ce_loss = F.cross_entropy(logits, labels)
        # cont_bias_feats = F.normalize(cont_bias_feats, dim=1)
        mask = 1 - cosine_similarity(cont_bias_feats.cpu().numpy())
        mask = torch.from_numpy(mask).cuda()
        con_loss = self.con_loss(cont_features, cont_labels, mask=mask)
        return ce_loss, con_loss


class SoftConTrainer(BaseTrainer):
    def _setup_models(self):
        self.model = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
            pretrained=self.cfg.MODEL.PRETRAINED,
        )
        self.model.to(self.device)

        bcc_net_dict = get_local_model_dict(self.cfg.MITIGATOR.SOFTCON.BCC_PATH)
        self.bcc_net = get_model(
            self.cfg.MODEL.TYPE,
            self.num_class,
        )
        self.bcc_net.load_state_dict(bcc_net_dict["model"])

        self.bcc_net.to(self.device)
        self.bcc_net.eval()

    def _train_iter(self, batch, cont_batch):
        inputs = batch["inputs"].to(self.device)
        targets = batch["targets"].to(self.device)

        cont_inputs = cont_batch["inputs"]
        cont_targets = cont_batch["targets"]
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        if isinstance(outputs, tuple):
            outputs, _ = outputs

        total_images = torch.cat([cont_inputs[0], cont_inputs[1]], dim=0)
        total_images, cont_labels = total_images.to(self.device), cont_targets.to(
            self.device
        )
        cont_inputs = cont_inputs[0]
        cont_inputs = cont_inputs.to(self.device)
        with torch.no_grad():
            _, cont_bias_feats = self.bcc_net(cont_inputs)

        _, cont_features = self.model(total_images)

        f1, f2 = torch.split(
            cont_features,
            [int(self.cfg.SOLVER.BATCH_SIZE / 2), int(self.cfg.SOLVER.BATCH_SIZE / 2)],
            dim=0,
        )
        cont_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        ce_loss, con_loss = self.criterion_train(
            outputs, targets, cont_features, cont_labels, cont_bias_feats
        )

        loss = ce_loss * self.cfg.MITIGATOR.SOFTCON.WEIGHT + con_loss

        self._loss_backward(loss)
        self._optimizer_step()
        self.scheduler.step()
        return {"train_cls_loss": ce_loss, "train_con_loss": con_loss}

    def _train_epoch(self):
        self._set_train()
        self.current_lr = self.scheduler.get_last_lr()[0]
        avg_loss = None
        for batch, cont_batch in zip(
            self.dataloaders["train"], self.dataloaders["train_cont"]
        ):
            bsz = batch["targets"].shape[0]
            loss_dict = self._train_iter(batch, cont_batch)
            # initialize if needed
            if avg_loss is None:
                avg_loss = {key: AverageMeter() for key in loss_dict.keys()}
            # Update avg_loss for each key in loss_dict
            for key, value in loss_dict.items():
                avg_loss[key].update(value.item(), bsz)
        avg_loss = {key: value.avg for key, value in avg_loss.items()}
        return avg_loss

    def create_cont_dataloader(self):
        if self.cfg.DATASET.TYPE == "urbancars":
            img_size = self.cfg.DATASET.URBANCARS.IMAGE_SIZE
        elif self.cfg.DATASET.TYPE == "waterbirds":
            img_size = self.cfg.DATASET.WATERBIRDS.IMAGE_SIZE
        elif self.cfg.DATASET.TYPE == "celeba":
            img_size = self.cfg.DATASET.CELEBA.IMAGE_SIZE
        elif self.cfg.DATASET.TYPE == "biased_mnist":
            img_size = self.cfg.DATASET.BIASED_MNIST.IMAGE_SIZE
        elif self.cfg.DATASET.TYPE == "fb_biased_mnist":
            img_size = self.cfg.DATASET.FB_BIASED_MNIST.IMAGE_SIZE
        elif self.cfg.DATASET.TYPE == "utkface":
            img_size = self.cfg.DATASET.UTKFACE.IMAGE_SIZE
        elif self.cfg.DATASET.TYPE == "imagenet9":
            img_size = self.cfg.DATASET.IMAGENET9.IMAGE_SIZE
        elif self.cfg.DATASET.TYPE == "stanford_dogs":
            img_size = self.cfg.DATASET.STANFORD_DOGS.IMAGE_SIZE
        elif self.cfg.DATASET.TYPE == "cifar10":
            img_size = self.cfg.DATASET.CIFAR10.IMAGE_SIZE
        elif self.cfg.DATASET.TYPE == "cifar100":
            img_size = self.cfg.DATASET.CIFAR100.IMAGE_SIZE
        else:
            raise NotImplementedError(
                f"You should define the image size for {self.cfg.DATASET.TYPE}"
            )
        transform = T.Compose(
            [
                T.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        transform = TwoCropTransform(transform)
        if self.cfg.DATASET.TYPE == "urbancars":
            cont_train_loader, _ = get_urbancars_loader(
                root=self.cfg.DATASET.URBANCARS.ROOT,
                batch_size=int(self.cfg.SOLVER.BATCH_SIZE / 2),
                image_size=self.cfg.DATASET.URBANCARS.IMAGE_SIZE,
                split="train",
                transform=transform,
            )
        elif self.cfg.DATASET.TYPE == "waterbirds":
            cont_train_loader, _ = get_waterbirds(
                self.cfg.DATASET.WATERBIRDS.ROOT,
                batch_size=int(self.cfg.SOLVER.BATCH_SIZE / 2),
                n_workers=self.cfg.DATASET.NUM_WORKERS,
                transform=transform,
                split="test",
            )
        elif self.cfg.DATASET.TYPE == "celeba":
            cont_train_loader, _ = get_celeba(
                self.cfg.DATASET.CELEBA.ROOT,
                batch_size=int(self.cfg.SOLVER.BATCH_SIZE / 2),
                split="train",
                target_attr=self.cfg.DATASET.CELEBA.TARGET,
                ratio=self.cfg.DATASET.CELEBA.RATIO,
                transform=transform,
            )
        elif self.cfg.DATASET.TYPE == "biased_mnist":
            cont_train_loader, _ = get_color_mnist(
                self.cfg.DATASET.BIASED_MNIST.ROOT,
                batch_size=int(self.cfg.SOLVER.BATCH_SIZE / 2),
                data_label_correlation=self.cfg.DATASET.BIASED_MNIST.CORR,
                n_confusing_labels=9,
                split="train",
                seed=self.cfg.EXPERIMENT.SEED,
                aug=False,
                transform=transform,
            )
        elif self.cfg.DATASET.TYPE == "fb_biased_mnist":
            cont_train_loader, _ = get_fb_color_mnist(
                self.cfg.DATASET.BIASED_MNIST.ROOT,
                batch_size=int(self.cfg.SOLVER.BATCH_SIZE / 2),
                data_label_correlation1=self.cfg.DATASET.FB_BIASED_MNIST.CORR_BG,
                data_label_correlation2=self.cfg.DATASET.FB_BIASED_MNIST.CORR_FG,
                n_confusing_labels=9,
                split="train",
                seed=self.cfg.EXPERIMENT.SEED,
                aug=False,
                transform=transform,
            )
        elif self.cfg.DATASET.TYPE == "utkface":
            cont_train_loader, _ = get_utk_face(
                self.cfg.DATASET.UTKFACE.ROOT,
                batch_size=int(self.cfg.SOLVER.BATCH_SIZE / 2),
                split="train",
                bias_attr=self.cfg.DATASET.UTKFACE.BIAS,
                ratio=self.cfg.DATASET.UTKFACE.RATIO,
                transform=transform,
            )
        elif self.cfg.DATASET.TYPE == "imagenet9":
            cont_train_loader, _ = get_imagenet9l(
                root=self.cfg.DATASET.IMAGENET9.ROOT_IMAGENET,
                batch_size=int(self.cfg.SOLVER.BATCH_SIZE / 2),
                image_size=self.cfg.DATASET.IMAGENET9.IMAGE_SIZE,
                transform=transform,
            )
        elif self.cfg.DATASET.TYPE == "stanford_dogs":
            cont_train_loader = get_stanford_dogs_loader(
                root=self.cfg.DATASET.STANFORD_DOGS.ROOT,
                batch_size=int(self.cfg.SOLVER.BATCH_SIZE / 2),
                image_size=self.cfg.DATASET.STANFORD_DOGS.IMAGE_SIZE,
                split="train",
                transform=transform,
            )
        elif self.cfg.DATASET.TYPE == "cifar10":
            cont_train_loader = get_cifar10_loaders(
                root=self.cfg.DATASET.CIFAR10.ROOT,
                batch_size=int(self.cfg.SOLVER.BATCH_SIZE / 2),
                image_size=self.cfg.DATASET.CIFAR10.IMAGE_SIZE,
                split="train",
                transform=transform,
            )
        elif self.cfg.DATASET.TYPE == "cifar100":
            cont_train_loader = get_cifar100_loaders(
                root=self.cfg.DATASET.CIFAR100.ROOT,
                batch_size=int(self.cfg.SOLVER.BATCH_SIZE / 2),
                image_size=self.cfg.DATASET.CIFAR100.IMAGE_SIZE,
                split="train",
                transform=transform,
            )
        else:
            raise NotImplementedError(
                f"You should create a contrastive dataloader for {self.cfg.DATASET.TYPE}"
            )

        self.dataloaders["train_cont"] = cont_train_loader

    def _method_specific_setups(self):
        self.create_cont_dataloader()

    def _setup_criterion(self):
        super()._setup_criterion()
        self.criterion_train = UnsupBiasContrastiveLoss()
