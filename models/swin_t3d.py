import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t
import torch.nn.functional as F


class SwinTransformer3D(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model=None):
        super().__init__()
        if model is None:
            if pretrained:
                weights = "Swin3D_T_Weights.KINETICS400_V1"
            else:
                weights = None
            model = swin3d_t(
                weights=weights
            )  # Swin Transformer Tiny model with pretrained weights
            self.extractor = model  # Remove classification head
            self.embed_size = (
                model.head.in_features
            )  # Get the embedding size from the final layer
            self.extractor.head = nn.Identity()
            self.num_classes = num_classes
            self.fc = nn.Linear(self.embed_size, num_classes)

        print(
            f"SwinTransformer Video - num_classes: {num_classes} pretrained: {pretrained}"
        )

    def forward(self, x, norm=False):
        feat = self.extractor(x)
        if norm:
            feat = F.normalize(feat, dim=1)
        logits = self.fc(feat)
        return logits, feat

    def badd_forward(self, x, f, m, norm=False):
        feat = self.extractor(x)
        if norm:
            feat = F.normalize(feat, dim=1)
        total_f = torch.sum(torch.stack(f), dim=0)
        feat = feat + total_f * m  # /2
        logits = self.fc(feat)
        return logits

    def mavias_forward(self, x, f, norm=False):
        feat = self.extractor(x)
        if norm:
            feat = F.normalize(feat, dim=1)
            f = F.normalize(f, dim=1)

        logits = self.fc(feat)
        logits2 = self.fc(f)

        return logits, logits2
