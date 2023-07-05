import torch
import torch.nn as nn


class SCSEModule(nn.Module):
    def __init__(self, in_channels, timesteps=5, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.timesteps = timesteps
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())
        # self.tSE = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),
        #     nn.Conv3d(timesteps, 1, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(1, timesteps, 1),
        #     nn.Softmax()
        # )

    def forward(self, x):
        h, w = x.size()[-2:]
        x = x * self.cSE(x) + x * self.sSE(x)
        # x = x.view(-1, self.timesteps, self.in_channels, h, w)
        # x = torch.sum(x * self.tSE(x), dim=1)
        return x