import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, width: int, cardinality: int):
        super(ResBlock, self).__init__()

        filters = width * cardinality

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, filters, 1, padding=0),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),

            nn.Conv2d(filters, filters, 3, padding=1, groups=cardinality),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),

            nn.Conv2d(filters, in_channels, 1, padding=0),
            nn.BatchNorm2d(in_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        path = self.conv(x)

        return self.relu(x + path)

class ResNeXt(nn.Module):
    def __init__(self, in_channels: int, size: int, n_classes: int, filters: int, blocks: int, width: int, cardinality: int):
        super(ResNeXt, self).__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(filters, width, cardinality) for _ in range(blocks)]
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(filters, 1, 1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),

            nn.Linear(size, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.res_blocks(x)

        return self.classifier(x)
