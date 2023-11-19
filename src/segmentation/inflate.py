import timm
import timm.models.layers as layers
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import convnext
from timm.models import _efficientnet_blocks


def inflate_conv2d(conv2d, temporal_dim=1):
    """Inflates a 2D convolution layer to 3D."""
    kh, kw = conv2d.kernel_size
    sh, sw = conv2d.stride
    ph, pw = conv2d.padding if not isinstance(conv2d, layers.Conv2dSame) else [kh // 2, kw // 2]
    dh, dw = conv2d.dilation
    groups = conv2d.groups
    conv3d = torch.nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_size=(temporal_dim, kh, kw),
        stride=(1, sh, sw),
        padding=(temporal_dim // 2, ph, pw),
        dilation=(1, dh, dw),
        groups=groups
    )
    weight_2d = conv2d.weight.data
    weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, temporal_dim, 1, 1) / temporal_dim
    conv3d.weight = nn.Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d


def inflate_bn2d(bn2d):
    """Inflates a 2D batch normalization layer to 3D."""
    bn3d = nn.BatchNorm3d(
        bn2d.num_features, 
        eps=bn2d.eps, 
        momentum=bn2d.momentum, 
        affine=bn2d.affine, 
        track_running_stats=bn2d.track_running_stats
    )
    bn3d.weight = bn2d.weight
    bn3d.bias = bn2d.bias
    return bn3d


def inflate_ln2d(ln2d):
    """Inflates a 2D layer normalization layer to 3D."""
    ln3d = nn.LayerNorm(
        ln2d.normalized_shape, 
        eps=ln2d.eps, 
        elementwise_affine=ln2d.elementwise_affine
    )
    ln3d.weight = ln2d.weight
    ln3d.bias = ln2d.bias
    return ln3d


def inflate_pool2d(pool2d, temporal_dim=1):
    """Inflates a 2D pooling layer to 3D."""
    if isinstance(pool2d, nn.MaxPool2d):
        pool3d = nn.MaxPool3d(
            (temporal_dim, pool2d.kernel_size, pool2d.kernel_size),
            stride=(1, pool2d.stride, pool2d.stride),
            padding=(temporal_dim // 2, pool2d.padding, pool2d.padding),
            dilation=(1, pool2d.dilation, pool2d.dilation),
            ceil_mode=pool2d.ceil_mode
        )
    elif isinstance(pool2d, nn.AvgPool2d):
        pool3d = nn.AvgPool3d(
            (temporal_dim, pool2d.kernel_size, pool2d.kernel_size),
            stride=(1, pool2d.stride, pool2d.stride),
            padding=(0, pool2d.padding, pool2d.padding)
        )
    else:
        raise ValueError(f"Layer {pool2d} is not supported.")
    return pool3d


class InflatedBatchNormAct2d(nn.Module):
    """Inflated BatchNormAct block."""
    def __init__(self, block, **kwargs):
        super().__init__(**kwargs)
        self.bn1 = inflate_bn2d(block)
        self.drop = block.drop
        self.act = block.act

    def forward(self, x):
        x = self.bn1(x)
        x = self.drop(x)
        x = self.act(x)
        return x


class InflatedLayerNorm(nn.Module):
    """Inflated LayerNorm module."""
    def __init__(self, block, **kwargs):
        super().__init__(**kwargs)
        self.ln3d = inflate_ln2d(block)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        weight, bias = self.ln3d.weight, self.ln3d.bias
        if torch.is_autocast_enabled():
            dt = torch.get_autocast_gpu_dtype()
            x, weight, bias = x.to(dt), weight.to(dt), bias.to(dt) if bias is not None else None
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x, self.ln3d.normalized_shape, weight, bias, self.ln3d.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x
        

class InflatedConvBnAct(nn.Module):
    """Inflated ConvBnAct block."""
    def __init__(self, block, temporal_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.conv = inflate_conv2d(block.conv, temporal_dim)
        self.bn1 = InflatedBatchNormAct2d(block.bn1)
        self.drop_path = block.drop_path
        self.has_skip = block.has_skip
    
    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x
    

class InflatedSqueezeExcite(nn.Module):
    """Inflated SqueezeExcite block."""
    def __init__(self, block, temporal_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.conv_reduce = inflate_conv2d(block.conv_reduce, temporal_dim)
        self.act1 = block.act1
        self.conv_expand = inflate_conv2d(block.conv_expand, temporal_dim)
        self.gate = block.gate

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

    
class InflatedEdgeResidual(nn.Module):
    """Inflated EdgeResidual block."""
    def __init__(self, block, temporal_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.conv_exp = inflate_conv2d(block.conv_exp, temporal_dim)
        self.bn1 = InflatedBatchNormAct2d(block.bn1)
        self.se = block.se
        self.conv_pwl = inflate_conv2d(block.conv_pwl, temporal_dim)
        self.bn2 = InflatedBatchNormAct2d(block.bn2)
        self.drop_path = block.drop_path
        self.has_skip = block.has_skip

    def forward(self, x):
        shortcut = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x

        
class InflatedInvertedResidual(nn.Module):
    """Inflated InvertedResidual block."""
    def __init__(self, block, temporal_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.conv_pw = inflate_conv2d(block.conv_pw, temporal_dim) # Only inflate the first point-wise convolution
        self.bn1 = InflatedBatchNormAct2d(block.bn1)
        self.conv_dw = inflate_conv2d(block.conv_dw, 1)
        self.bn2 = InflatedBatchNormAct2d(block.bn2)
        self.se = InflatedSqueezeExcite(block.se, 1)
        self.conv_pwl = inflate_conv2d(block.conv_pwl, 1)
        self.bn3 = InflatedBatchNormAct2d(block.bn3)
        self.drop_path = block.drop_path
        self.has_skip = block.has_skip

    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x
        

class InflatedConvNeXtBlock(nn.Module):
    """Inflated ConvNeXt block."""
    def __init__(self, block, temporal_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.conv_dw = inflate_conv2d(block.conv_dw, temporal_dim)
        self.ln1 = inflate_ln2d(block.norm)
        self.mlp = block.mlp
        self.gamma = block.gamma
        self.drop_path = block.drop_path
        self.shortcut = block.shortcut
        self.use_conv_mlp = block.use_conv_mlp
        
    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.ln1(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = self.ln1(x)
            x = self.mlp(x)
            x = x.permute(0, 4, 1, 2, 3)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class InflatedConvNeXtStage(nn.Module):
    """Inflated ConvNeXt stage."""
    def __init__(self, block, temporal_dim=1, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(block.downsample, nn.Identity):
            self.downsample = nn.Sequential(
                InflatedLayerNorm(block.downsample[0]),
                inflate_conv2d(block.downsample[1], 1)
            )
        else:
            self.downsample = nn.Identity()
        self.blocks = nn.Sequential(*[
            InflatedConvNeXtBlock(cnb, temporal_dim) for cnb in block.blocks
        ])

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x
    

class InflatedSplitAttn(nn.Module):
    """Inflated SplitAttn block."""
    def __init__(self, block, temporal_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.conv = inflate_conv2d(block.conv, temporal_dim)
        self.bn0 = inflate_bn2d(block.bn0)
        self.drop = block.drop
        self.act0 = block.act0
        self.fc1 = inflate_conv2d(block.fc1, temporal_dim)
        self.bn1 = inflate_bn2d(block.bn1)
        self.act1 = block.act1
        self.fc2 = inflate_conv2d(block.fc2, temporal_dim)
        self.rsoftmax = block.rsoftmax
        self.radix = block.radix

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.drop(x)
        x = self.act0(x)
        b, rc, t, h, w = x.shape
        if self.radix > 1:
            x = x.reshape((b, self.radix, rc // self.radix, t, h, w))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        x_gap = x_gap.mean((3, 4), keepdim=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)
        x_attn = self.rsoftmax(x_attn).view(b, -1, t, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((b, self.radix, rc // self.radix, t, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out.contiguous()


class InflatedResNestBottleneck(nn.Module):
    """Inflated ResNestBottleneck block."""
    def __init__(self, block, temporal_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = inflate_conv2d(block.conv1, temporal_dim)
        self.bn1 = inflate_bn2d(block.bn1)
        self.act1 = block.act1
        self.conv2 = InflatedSplitAttn(block.conv2, 1)
        self.bn2 = inflate_bn2d(block.bn2) if not isinstance(block.bn2, nn.Identity) else nn.Identity()
        self.drop_block = block.drop_block
        self.act2 = block.act2
        self.conv3 = inflate_conv2d(block.conv3, 1)
        self.bn3 = inflate_bn2d(block.bn3)
        self.act3 = block.act3
        self.downsample = None
        if block.downsample is not None:
            self.downsample = nn.Sequential(
                inflate_pool2d(block.downsample[0]) if not isinstance(block.downsample[0], nn.Identity) else nn.Identity(),
                inflate_conv2d(block.downsample[1], 1),
                inflate_bn2d(block.downsample[2])
            )
        self.avd_first = None
        if block.avd_first is not None:
            self.avd_first = inflate_pool2d(block.avd_first)
        self.avd_last = None
        if block.avd_last is not None:
            self.avd_last = inflate_pool2d(block.avd_last)

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        if self.avd_first is not None:
            out = self.avd_first(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_block(out)
        out = self.act2(out)
        if self.avd_last is not None:
            out = self.avd_last(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out += shortcut
        out = self.act3(out)
        return out
    
        
class InflatedEfficientNet(nn.Module):
    """Inflated EfficientNet model."""
    def __init__(
        self, 
        name, 
        backbone_params, 
        temporal_dim=1,
        block_idx=[4, 5],
        **kwargs
    ):
        super().__init__(**kwargs)
        encoder = timm.create_model(name, **backbone_params)
        self.temporal_dim = temporal_dim
        self.block_idx = block_idx
        self.conv_stem = inflate_conv2d(encoder.conv_stem, 1)
        self.bn1 = InflatedBatchNormAct2d(encoder.bn1)
        blocks = []
        blocks.append(
            nn.Sequential(*[
                InflatedConvBnAct(cba, 1) for cba in encoder.blocks[0]
            ])
        )
        blocks.append(
            nn.Sequential(*[
                InflatedEdgeResidual(er, 1) for er in encoder.blocks[1]
            ])
        )
        blocks.append(
            nn.Sequential(*[
                InflatedEdgeResidual(er, 1) for er in encoder.blocks[2]
            ])
        )
        for i in range(3, len(encoder.blocks)):
            blocks.append(
                nn.Sequential(*[
                    InflatedInvertedResidual(ir, temporal_dim if i in block_idx else 1) for ir in encoder.blocks[i]
                ])
            )
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.blocks(x)
        return x
    

class InflatedConvNeXt(nn.Module):
    """Inflated ConvNeXt model."""
    def __init__(
        self, 
        name, 
        backbone_params, 
        temporal_dim=1,
        block_idx=[2, 3],
        **kwargs
    ):
        super().__init__(**kwargs)
        encoder = timm.create_model(name, **backbone_params)
        self.temporal_dim = temporal_dim
        self.stem = nn.Sequential(
            inflate_conv2d(encoder.stem[0], 1),
            InflatedLayerNorm(encoder.stem[1])
        )
        stages = []
        for i in range(len(encoder.stages)):
            stages.append(
                InflatedConvNeXtStage(encoder.stages[i], temporal_dim if i in block_idx else 1)
            )
        self.stages = nn.Sequential(*stages)
        self.norm_pre = encoder.norm_pre
        
    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x
    

class InflatedResNest(nn.Module):
    """Inflated ResNest model."""
    def __init__(
        self, 
        name, 
        backbone_params, 
        temporal_dim=1,
        block_idx=[2, 3],
        **kwargs
    ):
        super().__init__(**kwargs)
        encoder = timm.create_model(name, **backbone_params)
        self.temporal_dim = temporal_dim
        self.conv1 = nn.Sequential(
            inflate_conv2d(encoder.conv1[0], 1),
            inflate_bn2d(encoder.conv1[1]),
            encoder.conv1[2],
            inflate_conv2d(encoder.conv1[3], 1),
            inflate_bn2d(encoder.conv1[4]),
            encoder.conv1[5],
            inflate_conv2d(encoder.conv1[6], 1),
        )
        self.bn1 = inflate_bn2d(encoder.bn1)
        self.act1 = encoder.act1
        self.maxpool = inflate_pool2d(encoder.maxpool)
        self.layer1 = nn.Sequential(*[
            InflatedResNestBottleneck(rnb, temporal_dim if 0 in block_idx else 1) for rnb in encoder.layer1
        ])
        self.layer2 = nn.Sequential(*[
            InflatedResNestBottleneck(rnb, temporal_dim if 1 in block_idx else 1) for rnb in encoder.layer2
        ])
        self.layer3 = nn.Sequential(*[
            InflatedResNestBottleneck(rnb, temporal_dim if 2 in block_idx else 1) for rnb in encoder.layer3
        ])
        self.layer4 = nn.Sequential(*[
            InflatedResNestBottleneck(rnb, temporal_dim if 3 in block_idx else 1) for rnb in encoder.layer4
        ])
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x