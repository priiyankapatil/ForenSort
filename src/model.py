"""ResNet-18 encoder + small decoder -> per-cell quality logits at 1/8 input resolution.

Input  (B, 1, S, S) normalised grayscale
Output (B, 6, S/8, S/8) class logits (levels 0..5), aligned with dataset targets
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

NUM_CLASSES = 6


def conv_bn(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class QualityNet(nn.Module):
    def __init__(self, pretrained=True, num_classes=NUM_CLASSES):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # 1-channel input: sum the RGB filters of the first conv so pretrained knowledge is kept
        w = net.conv1.weight.data.sum(dim=1, keepdim=True)
        net.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        net.conv1.weight.data = w
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)  # stride 4
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        # decoder: 32 -> 16 -> 8 (merge with encoder features at each step)
        self.up16 = conv_bn(512 + 256, 256)
        self.up8 = conv_bn(256 + 128, 128)
        self.head = nn.Sequential(conv_bn(128, 64), nn.Conv2d(64, num_classes, 1))

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)   # stride 8
        c3 = self.layer3(c2)   # stride 16
        c4 = self.layer4(c3)   # stride 32
        y = self.up16(torch.cat([F.interpolate(c4, size=c3.shape[-2:], mode="bilinear", align_corners=False), c3], 1))
        y = self.up8(torch.cat([F.interpolate(y, size=c2.shape[-2:], mode="bilinear", align_corners=False), c2], 1))
        return self.head(y)

    def freeze_encoder(self, frozen=True):
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            for p in m.parameters():
                p.requires_grad = not frozen


def masked_loss(
    logits,
    target,
    class_weight=None,
    ignore_index=255,
    ordinal_weight=0.0,
):
    """Classification loss plus an optional ordinal distance penalty."""
    ce = F.cross_entropy(logits, target, weight=class_weight, ignore_index=ignore_index)
    if ordinal_weight <= 0:
        return ce
    valid = target != ignore_index
    if not valid.any():
        return ce
    probabilities = logits.softmax(1)
    levels = torch.arange(NUM_CLASSES, device=logits.device, dtype=logits.dtype)
    expected = (probabilities * levels[None, :, None, None]).sum(1)
    distance = F.smooth_l1_loss(expected[valid], target[valid].to(logits.dtype))
    return ce + ordinal_weight * distance
