"""ResNet-20 for CIFAR-10 (He et al. 2016), option-A shortcuts (D1).

Quantisation points follow D9: the image is quantised once (fixed scale
1/255, D2), every block output is quantised once and used both as the next
conv input and as the next residual, and the pooled features are quantised
before the FC layer.
"""
import torch.nn as nn
import torch.nn.functional as F

from quant import periphery
from quant.qlayers import QConv2d, QLinear
from quant.quantizers import ActQuant


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.conv1 = QConv2d(in_planes, planes, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.q1 = ActQuant()
        self.conv2 = QConv2d(planes, planes, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.q_out = ActQuant()
        self.stride = stride
        self.pad = planes - in_planes

    def shortcut(self, x):
        # Option A: parameter-free subsample + zero-pad the extra channels.
        if self.stride == 1 and self.pad == 0:
            return x
        x = x[:, :, ::self.stride, ::self.stride]
        return F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))

    def forward(self, x):
        out = self.q1(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return self.q_out(out)


class ResNetCIFAR(nn.Module):
    def __init__(self, num_blocks=3, num_classes=10):
        super().__init__()
        self.q_in = ActQuant(fixed_scale=1.0 / 255.0)
        self.conv1 = QConv2d(3, 16, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.q1 = ActQuant()
        self.layer1 = self._make_layer(16, 16, num_blocks, stride=1)
        self.layer2 = self._make_layer(16, 32, num_blocks, stride=2)
        self.layer3 = self._make_layer(32, 64, num_blocks, stride=2)
        self.q_pool = ActQuant()
        self.fc = QLinear(64, num_classes)
        for m in self.modules():
            if isinstance(m, (QConv2d, QLinear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out" if isinstance(m, QConv2d) else "fan_in")
                nn.init.zeros_(m.bias)

    @staticmethod
    def _make_layer(in_planes, planes, n, stride):
        blocks = [BasicBlock(in_planes, planes, stride)]
        blocks += [BasicBlock(planes, planes, 1) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.q_in(x)
        x = self.q1(F.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.q_pool(periphery.global_avgpool(x))
        return self.fc(x)

    def crossbar_layers(self):
        """Ordered (name, layer, input-quantiser name) for every crossbar layer (D7, D14)."""
        out = [("conv1", self.conv1, "q_in")]
        prev = "q1"
        for li in (1, 2, 3):
            for bi, block in enumerate(getattr(self, f"layer{li}")):
                name = f"layer{li}.{bi}"
                out.append((f"{name}.conv1", block.conv1, prev))
                out.append((f"{name}.conv2", block.conv2, f"{name}.q1"))
                prev = f"{name}.q_out"
        out.append(("fc", self.fc, "q_pool"))
        return out


def resnet20(num_classes=10):
    return ResNetCIFAR(num_blocks=3, num_classes=num_classes)
