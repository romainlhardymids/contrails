import sys
import torch
import torch.nn as nn

from . import nextvit
from timm import create_model
from segmentation.inflate import InflatedEfficientNet, InflatedConvNeXt, InflatedResNest


def create_encoder(encoder_params):
    """Initializes an encoder from a given configuration."""
    module = getattr(sys.modules[__name__], encoder_params["class"])
    name = encoder_params["encoder_name"]
    return module(name=name, **encoder_params["params"])


class BaseEncoder(nn.Module):
    """Base encoder class."""
    def __init__(self, out_channels, **kwargs):
        super().__init__()
        self.out_channels = out_channels

    def get_stages(self):
        return [nn.Identity()]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for stage in stages:
            x = stage(x)
            features.append(x)
        return features


class ConvNeXtEncoder2d(BaseEncoder):
    """ConvNeXt encoder class (2D)."""
    def __init__(
        self, 
        name,
        stage_idx,
        backbone_params={},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = create_model(name, **backbone_params)
        assert len(stage_idx) <= len(self.encoder.stages)
        self.stage_idx = stage_idx
        self.depth = len(stage_idx) + 2

    def get_stages(self):
        return [nn.Identity(), self.encoder.stem] + \
            [self.encoder.stages[i : j] for i, j in zip([0] + self.stage_idx, self.stage_idx + [len(self.encoder.stages)])]
    

class ConvNeXtEncoder3d(BaseEncoder):
    """ConvNeXt encoder class (3D)."""
    def __init__(
        self, 
        name,
        stage_idx,
        backbone_params={},
        temporal_dim=1,
        block_idx=[5, 6],
        slice_idx=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = InflatedConvNeXt(name, backbone_params, temporal_dim, block_idx)
        assert slice_idx < temporal_dim
        assert len(stage_idx) <= len(self.encoder.stages)
        self.stage_idx = stage_idx
        self.depth = len(stage_idx) + 2
        self.slice_idx = slice_idx

    def get_stages(self):
        return [nn.Identity(), self.encoder.stem] + \
            [self.encoder.stages[i : j] for i, j in zip([0] + self.stage_idx, self.stage_idx + [len(self.encoder.stages)])]
    
    def forward(self, x):
        stages = self.get_stages()
        features = []
        for stage in stages:
            x = stage(x)
            features.append(x[:, :, self.slice_idx, :, :])
        return features


class EfficientNetEncoder2d(BaseEncoder):
    """EfficientNet encoder class (2D)."""
    def __init__(
        self, 
        name,
        stage_idx,
        backbone_params={},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = create_model(name, **backbone_params)
        assert len(stage_idx) <= len(self.encoder.blocks)
        self.stage_idx = stage_idx
        self.depth = len(stage_idx) + 2

    def get_stages(self):
        return [nn.Identity(), nn.Sequential(self.encoder.conv_stem, self.encoder.bn1)] + \
            [self.encoder.blocks[i : j] for i, j in zip([0] + self.stage_idx, self.stage_idx + [len(self.encoder.blocks)])]
    

class EfficientNetEncoder3d(BaseEncoder):
    """EfficientNet encoder class (3D)."""
    def __init__(
        self, 
        name,
        stage_idx,
        backbone_params={},
        temporal_dim=1,
        block_idx=[5, 6],
        slice_idx=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = InflatedEfficientNet(name, backbone_params, temporal_dim, block_idx)
        assert slice_idx < temporal_dim
        assert len(stage_idx) <= len(self.encoder.blocks)
        self.stage_idx = stage_idx
        self.depth = len(stage_idx) + 2
        self.slice_idx = slice_idx

    def get_stages(self):
        return [nn.Identity(), nn.Sequential(self.encoder.conv_stem, self.encoder.bn1)] + \
            [self.encoder.blocks[i : j] for i, j in zip([0] + self.stage_idx, self.stage_idx + [len(self.encoder.blocks)])]
    
    def forward(self, x):
        stages = self.get_stages()
        features = []
        for stage in stages:
            x = stage(x)
            features.append(x[:, :, self.slice_idx, :, :])
        return features


class MaxxViTEncoder2d(BaseEncoder):
    """MaxViT encoder class (2D)."""
    def __init__(
        self, 
        name,
        backbone_params={},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = create_model(name, **backbone_params)
        self.depth = 5

    def get_stages(self):
        return [
            nn.Identity(),
            self.encoder.stem,
            self.encoder.stages[0],
            self.encoder.stages[1],
            self.encoder.stages[2],
            self.encoder.stages[3]
        ]


class NextViTEncoder2d(BaseEncoder):
    """NextViT encoder class (2D)."""
    def __init__(
        self, 
        name, 
        stage_idx,
        create_fn,
        backbone_params={}, 
        **kwargs
    ):
        super().__init__(**kwargs)
        backbone_params = {} if backbone_params is None else backbone_params
        self.encoder = getattr(nextvit, create_fn)(**backbone_params)
        if "upernet" in name:
            state_dict = torch.load(f"../models/nextvit/{name}.pth")["state_dict"]
            state_dict = {
                k.replace("backbone.", ""): v
                for k, v in state_dict.items()
            }
        else:
            state_dict = torch.load(f"../models/nextvit/{name}.pth")["model"]
        self.encoder.load_state_dict(state_dict, strict=False)
        self.stage_idx = stage_idx

    def forward(self, x):
        features = []
        stage_id = 0
        features.append(x)
        x = self.encoder.stem(x)
        for i, layer in enumerate(self.encoder.features):
            x = layer(x)
            if i == self.stage_idx[stage_id]:
                if i == 0:
                    features.append(nn.functional.interpolate(x, scale_factor=2., mode="bilinear"))
                else:
                    features.append(x)
                stage_id += 1
        return features


class ResNetEncoder2d(BaseEncoder):
    """ResNet encoder class (2D)."""
    def __init__(
        self, 
        name,
        backbone_params={},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = create_model(name, **backbone_params)
        self.depth = 5

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.act1),
            nn.Sequential(self.encoder.maxpool, self.encoder.layer1),
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
        ]


class ResNestEncoder2d(BaseEncoder):
    """ResNest encoder class (2D)."""
    def __init__(
        self, 
        name,
        backbone_params={},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = create_model(name, **backbone_params)
        self.depth = 5

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.act1),
            nn.Sequential(self.encoder.maxpool, self.encoder.layer1),
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
        ]
    

class ResNestEncoder3d(BaseEncoder):
    """ResNest encoder class (3D)."""
    def __init__(
        self, 
        name,
        backbone_params={},
        temporal_dim=1,
        block_idx=[5, 6],
        slice_idx=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = InflatedResNest(name, backbone_params, temporal_dim, block_idx)
        assert slice_idx < temporal_dim
        self.slice_idx = slice_idx

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.act1),
            nn.Sequential(self.encoder.maxpool, self.encoder.layer1),
            self.encoder.layer2,
            self.encoder.layer3,
            self.encoder.layer4,
        ]
    
    def forward(self, x):
        stages = self.get_stages()
        features = []
        for stage in stages:
            x = stage(x)
            features.append(x[:, :, self.slice_idx, :, :])
        return features