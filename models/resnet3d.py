import torch.nn as nn

from torchvision.models.video import r3d_18

import torch.nn.functional as F
import torch


def set_resnet_fc(model, num_classes):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class VResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()

        model = r3d_18(pretrained=pretrained)
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.embed_size = 512
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes)

    def forward(self, x, norm=False):
        feat = self.extractor(x)
        feat = feat.view(feat.size(0), -1)
        if norm:
            feat = F.normalize(feat, dim=1)
        logits = self.fc(feat)
        return logits, feat

    def badd_forward(self, x, f, m, norm=False):
        x = self.extractor(x)
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
        total_f = torch.sum(torch.stack(f), dim=0)
        feat = feat + total_f * m  # /2
        logits = self.fc(feat)
        return logits

    def mavias_forward(self, x, f, norm=False):
        x = self.extractor(x)
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
            f = F.normalize(f, dim=1)

        logits = self.fc(feat)
        logits2 = self.fc(f)

        return logits, logits2


class VResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()

        # model = r3d_50(pretrained=pretrained)
        model = torch.hub.load(
            "facebookresearch/pytorchvideo", "slow_r50", pretrained=True
        )
        self.pool = nn.AvgPool3d(
            kernel_size=(8, 7, 7), stride=(1, 1, 1), padding=(0, 0, 0)
        )
        # modules = list(model.children()[0].children())[:-1]
        # print(modules)
        self.extractor = nn.Sequential(nn.Sequential(*model.blocks[:-1]))
        self.embed_size = 2048
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes)

    def forward(self, x, norm=False):
        feat = self.extractor(x)
        # print("After extractor:", feat.shape)
        feat = F.adaptive_avg_pool3d(feat, 1)
        # print("After pool:", feat.shape)
        feat = feat.view(feat.size(0), -1)

        if norm:
            feat = F.normalize(feat, dim=1)
        logits = self.fc(feat)
        return logits, feat

    def badd_forward(self, x, f, m, norm=False):
        x = self.extractor(x)
        feat = F.adaptive_avg_pool3d(feat, 1)
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
        total_f = torch.sum(torch.stack(f), dim=0)
        feat = feat + total_f * m  # /2
        logits = self.fc(feat)
        return logits

    def mavias_forward(self, x, f, norm=False):
        x = self.extractor(x)
        feat = F.adaptive_avg_pool3d(feat, 1)
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
            f = F.normalize(f, dim=1)

        logits = self.fc(feat)
        logits2 = self.fc(f)

        return logits, logits2
