import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(activation):
    """Returns the appropriate activation function given an alias."""
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "silu":
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f"Activation {activation} is not supported.")


class SeparableConv2d(nn.Sequential):
    """Separable 2D convolution layer."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        super().__init__(dephtwise_conv, pointwise_conv)


class SeparableConvBnAct(nn.Sequential):
    """Module consisting of a separable 2D convolution layer, a batch normalization layer, and an activation layer."""
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        use_batchnorm=True, 
        activation="silu"
    ):
        conv = SeparableConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=not (use_batchnorm),
        )
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        act = get_activation(activation)
        super(SeparableConvBnAct, self).__init__(conv, bn, act)


class ConvBnAct(nn.Sequential):
    """Module consisting of a normal 2D convolution layer, a batch normalization layer, and an activation layer."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
        activation=None
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        act = get_activation(activation)
        super(ConvBnAct, self).__init__(conv, bn, act)


class ASPPPooling(nn.Sequential):
    """Atrous Spatial Pyramid Pooling pooling layer."""
    def __init__(self, in_channels, out_channels, activation="silu"):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            ConvBnAct(in_channels, out_channels, kernel_size=1, activation=activation)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for module in self:
            x = module(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling block."""
    def __init__(
        self, 
        in_channels,
        out_channels, 
        atrous_rates, 
        reduction=1,
        dropout=0.2, 
        activation="silu"
    ):
        super(ASPP, self).__init__()
        modules = []
        modules.append(
            ConvBnAct(
                in_channels, 
                out_channels // reduction, 
                kernel_size=1, 
                padding=0,
                stride=1,
                use_batchnorm=True,
                activation=activation
            )
        )
        for r in atrous_rates:
            modules.append(
                SeparableConvBnAct(
                in_channels, 
                out_channels // reduction, 
                kernel_size=3,
                stride=1,
                padding=r,
                dilation=r,
                use_batchnorm=True,
                activation=activation
            ))
        modules.append(ASPPPooling(in_channels, out_channels // reduction, activation=activation))
        self.body = nn.ModuleList(modules)
        self.project = nn.Sequential(
            ConvBnAct(
                (len(atrous_rates) + 2) * out_channels // reduction, 
                out_channels, 
                kernel_size=1, 
                padding=0, 
                stride=1,
                use_batchnorm=True,
                activation=activation
            ),
            nn.Dropout(dropout)
        )

    def forward(self, x, scale_factor=1):
        if scale_factor != 1:
            x = F.interpolate(x, scale_factor=scale_factor, mode="bilinear")
        results = []
        for module in self.body:
            results.append(module(x))
        results = torch.cat(results, dim=1)
        return self.project(results)


class SegmentationHead(nn.Sequential):
    """Generic segmentation head."""
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3, 
        padding=1, 
        upsampling=1
    ):
        blocks = [
            ConvBnAct(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                use_batchnorm=False,
                activation=None
            ),
            nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        ]
        super(SegmentationHead, self).__init__(*blocks)


class SCSEModule(nn.Module):
    """Squeeze-channel and squeeze-excite module."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return (x * self.cSE(x) + x * self.sSE(x)) / 2.
    

class Attention(nn.Module):
    """UNet attention module."""
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention type {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)
    

class SegformerMLP(nn.Module):
    """Segformer MLP module."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    

class Stem3d(nn.Module):
    """Stem module for 3D segmentation models."""
    def __init__(self, in_channels, temporal_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, 
            out_channels=256, 
            kernel_size=(temporal_dim, 1, 1), 
            padding=(temporal_dim // 2, 0, 0),
            stride=(1, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(256)
        self.act1 = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv3d(
            in_channels=256, 
            out_channels=256,
            kernel_size=(1, 3, 3), 
            padding=(0, 1, 1),
            stride=(1, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(256)
        self.act2 = nn.SiLU(inplace=True)
        self.conv3 = nn.Conv3d(
            in_channels=256, 
            out_channels=in_channels,
            kernel_size=(1, 3, 3), 
            padding=(0, 1, 1),
            stride=(1, 1, 1)
        )
        self.bn3 = nn.BatchNorm3d(in_channels)
        self.act3 = nn.SiLU(inplace=True)

    def forward(self, x):
        assert len(x.size()) == 5
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        return x.max(dim=2)[0]