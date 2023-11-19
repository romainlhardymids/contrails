import torch.nn as nn
import transformers

from segmentation import decoder
from segmentation.encoder import create_encoder
from segmentation.modules import SegmentationHead, Stem3d
from segmentation_models_pytorch.base.initialization import initialize_decoder, initialize_head
from timm import create_model


def create_segmentation_model(config):
    config_ = config.copy()
    family = config_.pop("family")
    if family == "unet":
        return Unet(**config_)
    elif family == "upernet":
        return UperNet(**config_)
    elif family == "segformer":
        return Segformer(**config_)
    elif family == "dlv3":
        return DeepLabV3Plus(**config_)
    else:
        raise ValueError(f"Model family {family} is not supported.")
    

def create_classification_model(config):
    return ContrailsClassifier(**config)


class Unet(nn.Module):
    def __init__(
        self, 
        encoder_params,
        decoder_params, 
        num_classes=1
    ):
        super().__init__()
        # self.stem = nn.Identity() if not add_3d_stem else Stem3d(encoder_params["params"]["backbone_params"]["in_chans"], encoder_params["params"]["temporal_dim"])
        self.encoder = create_encoder(encoder_params)
        self.decoder = decoder.UnetDecoder(self.encoder.out_channels, **decoder_params)
        self.seg_head = SegmentationHead(
            decoder_params["decoder_channels"][-1], 
            num_classes, 
            kernel_size=3,
            padding=1, 
            upsampling=1
        )
        initialize_decoder(self.decoder)
        initialize_head(self.seg_head)

    def forward(self, x):
        # x = self.stem(x)
        # if len(x.size()) == 5:
        #     x = x[:, :, 0]
        x = x.squeeze(dim=2)
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        logits = self.seg_head(decoder_output)
        return logits


# class Segformer(nn.Module):
#     def __init__(
#         self, 
#         encoder_name, 
#         encoder_params,
#         decoder_params, 
#         num_classes=1,
#         use_2d=True
#     ):
#         super().__init__()
#         self.encoder = create_encoder(encoder_name, encoder_params, use_2d=use_2d)
#         self.decoder = decoder.SegformerDecoder(self.encoder.out_channels, **decoder_params)
#         self.seg_head = SegmentationHead(
#             decoder_params["hidden_size"], 
#             num_classes, 
#             kernel_size=3,
#             padding=1, 
#             upsampling=2
#         )
#         initialize_decoder(self.decoder)
#         initialize_head(self.seg_head)

#     def forward(self, x):
#         features = self.encoder(x)
#         decoder_output = self.decoder(features[1:])
#         seg_logits = self.seg_head(decoder_output)
#         return seg_logits


class Segformer(nn.Module):
    def __init__(self, encoder_params):
        super().__init__()
        # self.stem = nn.Identity() if not add_3d_stem else Stem3d(encoder_params["params"]["backbone_params"]["num_channels"], encoder_params["params"]["temporal_dim"])
        self.backbone = transformers.SegformerForSemanticSegmentation.from_pretrained(encoder_params["encoder_name"], **encoder_params["params"]["backbone_params"])
        self.upscaler = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, x):
        # x = self.stem(x)
        # if len(x.size()) == 5:
        #     x = x[:, :, 0]
        x = x.squeeze(dim=2)
        x = self.backbone(x).logits
        x = self.upscaler(x)
        return x
        


class UperNet(nn.Module):
    def __init__(
        self, 
        encoder_name, 
        encoder_params,
        decoder_params, 
        num_classes=1,
        use_2d=True
    ):
        super().__init__()
        self.encoder = create_encoder(encoder_name, encoder_params, use_2d=use_2d)
        self.decoder = decoder.UperNetDecoder(self.encoder.out_channels, **decoder_params)
        self.seg_head = SegmentationHead(
            decoder_params["hidden_size"], 
            num_classes, 
            kernel_size=3,
            padding=1, 
            upsampling=2
        )
        initialize_decoder(self.decoder)
        initialize_head(self.seg_head)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(features[1:])
        seg_logits = self.seg_head(decoder_output)
        return seg_logits


class DeepLabV3Plus(nn.Module):
    def __init__(
        self, 
        encoder_name, 
        encoder_params,
        decoder_params, 
        num_classes=1,
        upsampling=4,
        use_2d=True
    ):
        super().__init__()
        self.encoder = create_encoder(encoder_name, encoder_params, use_2d=use_2d)
        self.decoder = decoder.DeepLabV3PlusDecoder(self.encoder.out_channels, **decoder_params)
        self.seg_head = SegmentationHead(
            decoder_params["decoder_channels"], 
            num_classes, 
            kernel_size=1, 
            padding=0, 
            upsampling=upsampling
        )
        initialize_decoder(self.decoder)
        initialize_head(self.seg_head)

    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        seg_logits = self.seg_head(decoder_output)
        return seg_logits
    

class ContrailsClassifier(nn.Module):
    def __init__(
        self, 
        encoder_name, 
        representation_dim, 
        dropout=0.2, 
        backbone_params={}, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = create_model(encoder_name, num_classes=0, **backbone_params)
        self.drop = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.clf_head = nn.Linear(self.model.num_features, 1)
        self.rep_head = nn.Linear(self.model.num_features, representation_dim)
    
    def forward(self, x):
        if len(x.size()) == 5:
            x = x[:, :, 0]
        x = self.model(x)
        x = self.drop(x)
        clf_logits = self.clf_head(x)[:, 0]
        rep_logits = self.rep_head(x)
        return clf_logits, rep_logits