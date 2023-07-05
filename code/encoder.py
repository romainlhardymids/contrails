import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from modules import SCSEModule
from nextvit import nextvit_base
from segment_anything.modeling import image_encoder
from segment_anything.modeling.common import LayerNorm2d
from timm import create_model


class ResNestEncoder(nn.Module):
    def __init__(self, backbone, depth=5, timesteps=5, **kwargs):
        super().__init__()
        self.model = create_model(backbone, pretrained=True, **kwargs)
        self._name = backbone
        self._depth = depth
        self._timesteps = timesteps
        del self.model.fc
        del self.model.global_pool

    @property
    def output_stride(self):
        return 32
    
    @property
    def _out_channels(self):
        return [c * self._timesteps for c in encoder_configs[self._name]["out_channels"]]
    
    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.model.conv1, self.model.bn1, self.model.act1),
            nn.Sequential(self.model.maxpool, self.model.layer1),
            self.model.layer2,
            self.model.layer3,
            self.model.layer4
        ]

    def forward(self, x):
        c, h, w = x.size()[-3:]
        x = x.view(-1, c, h, w)
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x.view(-1, self._out_channels[i], h // 2 ** i, w // 2 ** i))
        return features
    

class EfficientNetEncoder(nn.Module):
    def __init__(self, backbone, depth=5, timesteps=5, **kwargs):
        super().__init__()
        self.model = create_model(backbone, pretrained=True, **kwargs)
        self._name = backbone
        self._stage_idxs = [2, 3, 5]
        self._depth = depth
        self._timesteps = timesteps
        del self.model.classifier

    @property
    def output_stride(self):
        return 32
    
    @property
    def _out_channels(self):
        return [c * self._timesteps for c in encoder_configs[self._name]["out_channels"]]

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.model.conv_stem, self.model.bn1),
            self.model.blocks[: self._stage_idxs[0]],
            self.model.blocks[self._stage_idxs[0] : self._stage_idxs[1]],
            self.model.blocks[self._stage_idxs[1] : self._stage_idxs[2]],
            self.model.blocks[self._stage_idxs[2] :],
        ]

    def forward(self, x):
        c, h, w = x.size()[-3:]
        x = x.view(-1, c, h, w)
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x.view(-1, self._out_channels[i], h // 2 ** i, w // 2 ** i))
        return features
    


class ConvNeXtEncoder(nn.Module):
    def __init__(self, backbone, depth=5, timesteps=5, **kwargs):
        super().__init__()
        self.model = create_model(backbone, pretrained=True, **kwargs)
        self._name = backbone
        self._depth = depth
        self._timesteps = timesteps


    @property
    def output_stride(self):
        return 32
    
    @property
    def _out_channels(self):
        return [c * self._timesteps for c in encoder_configs[self._name]["out_channels"]]

    def get_stages(self):
        return [
            nn.Identity(),
            self.model.stem,
            self.model.stages[0],
            self.model.stages[1],
            self.model.stages[2],
            self.model.stages[3]
        ]

    def forward(self, x):
        c, h, w = x.size()[-3:]
        x = x.view(-1, c, h, w)
        stages = self.get_stages()
        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            if i == 1:
                features.append(nn.functional.interpolate(x, scale_factor=2., mode="bilinear").view(-1, self._out_channels[i], h // 2 ** i, w // 2 ** i))
            else:
                features.append(x.view(-1, self._out_channels[i], h // 2 ** i, w // 2 ** i))
        return features
    

class NextViTEncoder(nn.Module):
    def __init__(self, backbone, depth=4, **kwargs):
        super().__init__()
        self.model = nextvit_base(**kwargs)
        self.model.load_state_dict(torch.load("../models/nextvit/nextvit_base_in1k_384.pth")["model"])
        self.model.stage_out_idx = [0, 2, 6, 26, 29]
        self._depth = depth

    @property
    def output_stride(self):
        return 32
    
    @property
    def _out_channels(self):
        return [3, 96, 96, 256, 512, 1024]

    def forward(self, x):
        features = []
        stage_id = 0
        features.append(x)
        x = self.model.stem(x)
        for i, layer in enumerate(self.model.features):
            if self.model.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
            if i == self.model.stage_out_idx[stage_id]:
                if i == 0:
                    features.append(nn.functional.interpolate(x, scale_factor=2., mode="bilinear"))
                else:
                    features.append(x)
                stage_id += 1
        return features
    

class SAMEncoder(nn.Module):
    # Source: https://github.com/Rusteam/segmentation_models.pytorch/blob/sam/segmentation_models_pytorch/encoders/sam.py
    def __init__(self, image_size, depth=4, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.vit_depth = 12
        self.encoder_depth = depth
        self.embed_dim = 768
        self.patch_size = 16
        self.vit = image_encoder.ImageEncoderViT(
            img_size=image_size,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            use_abs_pos=False,
            global_attn_indexes=[2, 5, 8, 11]
        )
        state_dict = torch.load("../models/sam-encoders/sam_vit_b_01ec64.pth")
        state_dict = {
            k.replace("image_encoder.", ""): v
            for k, v in state_dict.items()
            if not k.startswith("mask_decoder") and not k.startswith("prompt_encoder")
        }
        self.vit.load_state_dict(state_dict, strict=False)
        self.necks = nn.ModuleList(
            [self.init_neck(self.embed_dim, c) for c in self._out_channels[:-1]]
        )

    @staticmethod
    def init_neck(embed_dim, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
        )
    
    @staticmethod
    def neck_forward(neck, x, scale_factor=1.):
        x = x.permute(0, 3, 1, 2)
        if scale_factor != 1.0:
            x = nn.functional.interpolate(x, scale_factor=scale_factor, mode="bilinear")
        return neck(x)

    @property
    def output_stride(self):
        return 32

    @property
    def _out_channels(self):
        return [256 // (2 ** i) for i in range(self.encoder_depth + 1)][::-1]
    
    @property
    def scale_factor(self):
        return int(math.log(self.patch_size, 2))

    def forward(self, x):
        x = self.vit.patch_embed(x)
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed

        features = []
        skip_steps = self.vit_depth // self.encoder_depth
        scale_factor = self.scale_factor
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            if i % skip_steps == 0:
                neck = self.necks[i // skip_steps]
                features.append(self.neck_forward(neck, x, scale_factor=2 ** scale_factor))
                scale_factor -= 1

        x = self.vit.neck(x.permute(0, 3, 1, 2))
        features.append(x)

        return features

    

encoder_configs = {
    "tf_efficientnetv2_s": {
        "out_channels": (3, 24, 48, 64, 160, 256),
        "output_stride": 32
    },
    "tf_efficientnetv2_m.in21k_ft_in1k": {
        "out_channels": (3, 24, 48, 80, 176, 512),
        "output_stride": 32
    },
    "tf_efficientnetv2_l.in21k_ft_in1k": {
        "out_channels": (3, 32, 64, 96, 224, 640),
        "output_stride": 32
    },
    "convnext_large_384_in22ft1k": {
        "out_channels": (3, 192, 192, 384, 768, 1536),
        "output_stride": 32
    },
    "convnextv2_base.fcmae_ft_in22k_in1k_384": {
        "out_channels": (3, 128, 128, 256, 512, 1024),
        "output_stride": 32
    },
    "resnest101e.in1k": {
        "out_channels": (3, 128, 256, 512, 1024, 2048),
        "output_stride": 32
    }
}