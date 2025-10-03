import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=False, **kwargs):
        super(SimpleConvNet, self).__init__()
        kernel_size = 7
        padding = kernel_size // 2

        layers = [
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        self.extracter = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.dim_in = 128

        print(f"SimpleConvNet: kernel_size {kernel_size}")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_last_shared_layer(self):
        return self.fc

    def forward(self, x, norm=False):
        x = self.extracter(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
        logits = self.fc(feat)
        return logits, feat

    def badd_forward(self, x, f, m, norm=False):
        x = self.extracter(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
        total_f = torch.sum(torch.stack(f), dim=0)
        feat = feat + total_f * m  # /2
        logits = self.fc(feat)
        return logits

    def mavias_forward(self, x, f, norm=False):
        x = self.extracter(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
            f = F.normalize(f, dim=1)

        logits = self.fc(feat)
        logits2 = self.fc(f)

        return logits, logits2


class SimpleConvNetMultiHead(nn.Module):
    def __init__(self, num_classes=10, pretrained=False, **kwargs):
        super(SimpleConvNetMultiHead, self).__init__()
        kernel_size = 7
        padding = kernel_size // 2

        layer1 = [
            nn.Conv2d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ]
        layer2 = [
            nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        layer3 = [
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        layer4 = [
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        self.layer1 = nn.Sequential(*layer1)
        self.layer2 = nn.Sequential(*layer2)
        self.layer3 = nn.Sequential(*layer3)
        self.layer4 = nn.Sequential(*layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dim_in = 128

        self.fc1 = nn.Linear(16, num_classes)
        self.fc2 = nn.Linear(32, num_classes)
        self.fc3 = nn.Linear(64, num_classes)
        self.fc4 = nn.Linear(128, num_classes)

        self.pr1 = nn.Linear(768, 16)
        self.pr2 = nn.Linear(768, 32)
        self.pr3 = nn.Linear(768, 64)
        self.pr4 = nn.Linear(768, 128)

        print(f"SimpleConvNet: kernel_size {kernel_size}")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, norm=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
        logits = self.fc4(feat)
        return logits, feat

    def mavias_forward(self, x, f, norm=False):

        x = self.layer1(x)
        feat1 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        logits1 = self.fc1(feat1)
        clogits1 = self.fc1(self.pr1(f))

        x = self.layer2(x)
        feat2 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        logits2 = self.fc2(feat2)
        clogits2 = self.fc2(self.pr2(f))

        x = self.layer3(x)
        feat3 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        logits3 = self.fc3(feat3)
        clogits3 = self.fc3(self.pr3(f))

        x = self.layer4(x)
        # feat5 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        feat4 = self.avgpool(x)
        feat4 = feat4.squeeze(-1).squeeze(-1)
        logits4 = self.fc4(feat4)
        clogits4 = self.fc4((self.pr4(f)))

        return [logits1, logits2, logits3, logits4], [
            clogits1,
            clogits2,
            clogits3,
            clogits4,
        ]
