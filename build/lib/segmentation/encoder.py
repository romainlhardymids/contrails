import sys
import torch
import torch.nn as nn

from . import nextvit
from timm import create_model
from segmentation.inflate import InflatedEfficientNet, InflatedConvNeXt, InflatedResNest


def create_encoder(encoder_params):
    module = getattr(sys.modules[__name__], encoder_params["class"])
    name = encoder_params["encoder_name"]
    return module(name=name, **encoder_params["params"])


class BaseEncoder(nn.Module):
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


# ENCODER_CONFIGS = {
#     "convnextv2_tiny.fcmae_ft_in22k_in1k_384": {
#         "class": {
#             "2d": ConvNeXtEncoder2d,
#             "3d": ConvNeXtEncoder3d
#         },
#         "params": {
#             "output_stride": 32,
#             "out_channels": [3, 96, 96, 192, 384, 768],
#             "stage_idx": [1, 2, 3],
#             "backbone_params": {
#                 "pretrained": True,
#                 "drop_path_rate": 0.2
#             },
#             "temporal_dim": 3,
#             "slice_idx": 2,
#             "block_idx": [0, 1, 2, 3]
#         }
#     },
#     "convnextv2_base.fcmae_ft_in22k_in1k_384": {
#         "class": {
#             "2d": ConvNeXtEncoder2d,
#             "3d": ConvNeXtEncoder3d
#         },
#         "params": {
#             "output_stride": 32,
#             "out_channels": [3, 128, 128, 256, 512, 1024],
#             "stage_idx": [1, 2, 3],
#             "backbone_params": {
#                 "pretrained": True,
#                 "drop_path_rate": 0.2
#             },
#             "temporal_dim": 3,
#             "slice_idx": 2,
#             "block_idx": [0, 1, 2, 3]
#         }
#     },
#     "convnextv2_large.fcmae_ft_in22k_in1k_384": {
#         "class": {
#             "2d": ConvNeXtEncoder2d,
#             "3d": ConvNeXtEncoder3d,
#         },
#         "params": {
#             "output_stride": 32,
#             "out_channels": [3, 192, 192, 384, 768, 1536],
#             "stage_idx": [1, 2, 3],
#             "backbone_params": {
#                 "pretrained": True,
#                 "drop_path_rate": 0.3
#             },
#             "temporal_dim": 3,
#             "slice_idx": 2,
#             "block_idx": [0, 1, 2, 3]
#         }
#     },
#     "convnextv2_huge.fcmae_ft_in22k_in1k_512": {
#         "class": {
#             "2d": ConvNeXtEncoder2d
#         },
#         "params": {
#             "output_stride": 32,
#             "out_channels": [3, 352, 352, 704, 1408, 2816],
#             "stage_idx": [1, 2, 3],
#             "backbone_params": {
#                 "pretrained": True,
#                 "drop_path_rate": 0.2
#             }
#         }
#     },
#     "maxxvitv2_rmlp_base_rw_384.sw_in12k_ft_in1k": {
#         "class": {
#             "2d": MaxxViTEncoder2d
#         },
#         "params": {
#             "output_stride": 32,
#             "out_channels": [3, 128, 128, 256, 512, 1024],
#             "backbone_params": {
#                 "pretrained": True,
#                 "drop_path_rate": 0.2
#             }
#         }
#     },
#     "resnest101e.in1k": {
#         "class": {
#             "2d": ResNestEncoder2d,
#             "3d": ResNestEncoder3d
#         },
#         "params": {
#             "output_stride": 32,
#             "out_channels": [3, 128, 256, 512, 1024, 2048],
#             "backbone_params": {
#                 "pretrained": True
#             },
#             "temporal_dim": 3,
#             "slice_idx": 2,
#             "block_idx": [2, 3]
#         }
#     },
#     "upernet_160k_nextvit_large_1n1k6m_pretrained": {
#         "class": {
#             "2d": NextViTEncoder2d
#         },
#         "params": {
#             "output_stride": 32,
#             "out_channels": [3, 96, 256, 512, 1024],
#             "stage_idx": [2, 6, 36, 39],
#             "create_fn": nextvit_large,
#             "backbone_params": {}
#         }
#     },
#     "tf_efficientnetv2_s.in21k_ft_in1k": {
#         "class": {
#             "2d": EfficientNetEncoder2d,
#             "3d": EfficientNetEncoder3d
#         },
#         "params": {
#             "output_stride": 32,
#             "out_channels": [3, 24, 48, 64, 160, 256],
#             "stage_idx": [2, 3, 5],
#             "backbone_params": {
#                 "pretrained": True,
#                 "drop_path_rate": 0.2
#             },
#             "temporal_dim": 3,
#             "slice_idx": 2,
#             "block_idx": [4, 5]
#         }
#     },
#     "tf_efficientnetv2_m.in21k_ft_in1k": {
#         "class": {
#             "2d": EfficientNetEncoder2d,
#             "3d": EfficientNetEncoder3d
#         },
#         "params": {
#             "output_stride": 32,
#             "out_channels": [3, 24, 48, 80, 176, 512],
#             "stage_idx": [2, 3, 5],
#             "backbone_params": {
#                 "pretrained": True,
#                 "drop_path_rate": 0.2
#             },
#             "temporal_dim": 3,
#             "slice_idx": 2,
#             "block_idx": [5, 6]
#         }
#     },
#     "tf_efficientnetv2_l.in21k_ft_in1k": {
#         "class": {
#             "2d": EfficientNetEncoder2d,
#             "3d": EfficientNetEncoder3d
#         },
#         "params": {
#             "output_stride": 32,
#             "out_channels": [3, 32, 64, 96, 224, 640],
#             "stage_idx": [2, 3, 5],
#             "backbone_params": {
#                 "pretrained": True,
#                 "drop_path_rate": 0.2
#             },
#             "temporal_dim": 3,
#             "slice_idx": 2,
#             "block_idx": [5, 6]
#         }
#     },
#     "tf_efficientnetv2_xl.in21k_ft_in1k": {
#         "class": {
#             "2d": EfficientNetEncoder2d,
#             "3d": EfficientNetEncoder3d
#         },
#         "params": {
#             "output_stride": 32,
#             "out_channels": [3, 32, 64, 96, 256, 640],
#             "stage_idx": [2, 3, 5],
#             "backbone_params": {
#                 "pretrained": True,
#                 "drop_path_rate": 0.3
#             },
#             "temporal_dim": 3,
#             "slice_idx": 2,
#             "block_idx": [5, 6]
#         }
#     },
# }