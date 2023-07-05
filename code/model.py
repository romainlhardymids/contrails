from encoder import ConvNeXtEncoder, EfficientNetEncoder, NextViTEncoder, ResNestEncoder, SAMEncoder
from segmentation_models_pytorch.base import (
    ClassificationHead,
    SegmentationHead, 
    SegmentationModel
)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class Unet(SegmentationModel):
    def __init__(
        self,
        encoder_name="resnest101e.in1k",
        decoder_use_batchnorm=True,
        decoder_channels=[256, 128, 64, 32, 16],
        decoder_attention_type=None,
        classes=1,
        activation=None,
        aux_params=None,
        timesteps=5
    ):
        super().__init__()
        # self.encoder = Encoder(encoder_name, depth=5, stage_idxs=(2, 3, 5))
        # self.encoder = SAMEncoder(384, depth=4)
        self.encoder = ConvNeXtEncoder(encoder_name, depth=5, timesteps=timesteps)
        # self.encoder = NextViTEncoder(encoder_name, depth=5)
        # self.encoder = EfficientNetEncoder(encoder_name, timesteps=timesteps, depth=5)
        # self.encoder = ResNestEncoder(encoder_name, timesteps=timesteps)
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder._out_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder._out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = "u-{}".format(encoder_name)
        self.initialize()