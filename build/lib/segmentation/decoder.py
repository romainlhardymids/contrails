import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.models.upernet.modeling_upernet as upernet

from segmentation import modules as md


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        block_depth=1,
        separable=False,
        use_aspp=False,
        use_batchnorm=True,
        attention_type=None,
        activation="relu"
    ):
        super().__init__()
        self.attention = nn.ModuleList([
            md.Attention(attention_type, in_channels=in_channels + skip_channels),
            md.Attention(attention_type, in_channels=out_channels)
        ])
        self.aspp = md.ASPP(
            in_channels,
            in_channels,
            atrous_rates=[1, 2, 4],
            reduction=2,
            dropout=0.2,
            activation=activation
        ) if use_aspp else nn.Identity()
        module = md.SeparableConvBnAct if separable else md.ConvBnAct
        self.stem = module(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            activation=activation
        )
        self.body = nn.Sequential(*[
            module(
                out_channels, 
                out_channels, 
                kernel_size=3, 
                padding=1, 
                use_batchnorm=use_batchnorm,
                activation=activation
            ) for _ in range(block_depth)
         ])

    def forward(self, x, skip=None, scale_factor=1):
        if scale_factor != 1:
            x = F.interpolate(x, scale_factor=scale_factor, mode="bilinear")
        x = self.aspp(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention[0](x)
        x = self.stem(x)
        x = self.body(x)
        x = self.attention[1](x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        scale_factors,
        num_blocks=5,
        block_depth=1,
        separable=False,
        use_aspp=False,
        use_batchnorm=True,
        attention_type=None,
        activation="relu"
    ):
        super().__init__()
        assert num_blocks >= len(encoder_channels) - 1
        assert num_blocks == len(decoder_channels)
        assert num_blocks == len(scale_factors)
        self.scale_factors = scale_factors
        encoder_channels = encoder_channels[1:][::-1]
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:])
        skip_channels += [0] * (len(in_channels) - len(skip_channels))
        out_channels = decoder_channels
        aspp_idx = len(in_channels) - 2
        blocks = []
        for i, (i_ch, s_ch, o_ch) in enumerate(zip(in_channels, skip_channels, out_channels)):
            blocks.append(
                DecoderBlock(
                    i_ch, 
                    s_ch, 
                    o_ch, 
                    block_depth,
                    separable=separable,
                    use_aspp=use_aspp if i == aspp_idx else False,
                    use_batchnorm=use_batchnorm, 
                    attention_type=attention_type,
                    activation=activation
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:][::-1]
        x = features[0]
        skips = features[1:]
        for i, (block, scale_factor) in enumerate(zip(self.blocks, self.scale_factors)):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip, scale_factor)
        return x
    

class UperNetDecoder(nn.Module):
    def __init__(
        self, 
        encoder_channels,
        pool_scales,
        hidden_size,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_channels = encoder_channels[1:]
        self.pool_scales = pool_scales
        self.channels = hidden_size
        self.align_corners = False
        self.psp_modules = upernet.UperNetPyramidPoolingModule(
            self.pool_scales,
            self.encoder_channels[-1],
            self.channels,
            align_corners=self.align_corners
        )
        self.bottleneck = upernet.UperNetConvModule(
            self.encoder_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1
        )
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for e_ch in self.encoder_channels[:-1]: 
            lateral_conv = upernet.UperNetConvModule(e_ch, self.channels, kernel_size=1)
            fpn_conv = upernet.UperNetConvModule(self.channels, self.channels, kernel_size=3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
        self.fpn_bottleneck = upernet.UperNetConvModule(
            len(self.encoder_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1
        )

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output
    
    def forward(self, encoder_hidden_states):
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(encoder_hidden_states))
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], 
                size=prev_shape, 
                mode="bilinear", 
                align_corners=self.align_corners
            )
        fpn_outputs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        fpn_outputs.append(laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outputs[i] = nn.functional.interpolate(
                fpn_outputs[i], 
                size=fpn_outputs[0].shape[2:], 
                mode="bilinear", 
                align_corners=self.align_corners
            )
        fpn_outputs = torch.cat(fpn_outputs, dim=1)
        output = self.fpn_bottleneck(fpn_outputs)
        return output
    

class SegformerDecoder(nn.Module):
    def __init__(
        self, 
        encoder_channels, 
        hidden_size,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder_channels = encoder_channels[1:]
        mlps = []
        for i in range(len(self.encoder_channels)):
            mlp = md.SegformerMLP(self.encoder_channels[i], hidden_size)
            mlps.append(mlp)
        self.mlps = nn.ModuleList(mlps)
        self.fuse = md.ConvBnAct(
            in_channels=hidden_size * len(mlps),
            out_channels=hidden_size,
            kernel_size=1,
            use_batchnorm=True,
            activation="silu"
        )

    def forward(self, encoder_hidden_states):
        bsz = encoder_hidden_states[-1].shape[0]
        all_hidden_states = []
        for x, mlp in zip(encoder_hidden_states, self.mlps):
            h, w = x.shape[2], x.shape[3]
            x = mlp(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(bsz, -1, h, w)
            x = nn.functional.interpolate(
                x, 
                size=encoder_hidden_states[0].size()[2:], 
                mode="bilinear", 
                align_corners=False
            )
            all_hidden_states.append(x)
        output = self.fuse(torch.cat(all_hidden_states[::-1], dim=1))
        return output


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        block_depth=1,
        atrous_rates=[1, 2, 4],
        reduction=1,
        dropout=0.2,
        activation="relu",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.aspp = md.ASPP(
            encoder_channels[-1], 
            decoder_channels, 
            atrous_rates, 
            reduction=reduction, 
            dropout=dropout
        )
        self.low_res_blocks = nn.Sequential(*[
            md.ConvBnAct(
                decoder_channels, 
                decoder_channels, 
                kernel_size=3, 
                padding=1, 
                use_batchnorm=True,
                activation=activation
            ) for _ in range(block_depth)
        ])
        high_res_channels = encoder_channels[2]
        self.high_res_blocks = nn.Sequential(*[
            md.ConvBnAct(
                high_res_channels, 
                high_res_channels, 
                kernel_size=3, 
                padding=1, 
                use_batchnorm=True,
                activation=activation
            ) for _ in range(block_depth)
        ])
        self.head = md.ConvBnAct(
            decoder_channels + high_res_channels, 
            decoder_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            use_batchnorm=True,
            activation=activation
        )

    def get_scale_factor(self, features):
        h1 = features[2].shape[-2]
        h2 = features[-1].shape[-2]
        scale_factor = 2 ** (np.log2(h1 / h2) - 1.)
        return scale_factor

    def forward(self, *features):
        scale_factor = self.get_scale_factor(features)
        x1 = self.aspp(features[-1], scale_factor=2)
        x1 = self.low_res_blocks(x1)
        x1 = F.interpolate(x1, scale_factor=scale_factor) # Upsample to match high-resolution features
        x2 = self.high_res_blocks(features[2])
        x = torch.cat([x1, x2], dim=1)
        x = self.head(x)
        return x