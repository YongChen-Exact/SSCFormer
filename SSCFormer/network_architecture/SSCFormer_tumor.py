from torch import nn
import torch

from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from SSCFormer.network_architecture.layers import LayerNorm
from SSCFormer.network_architecture.dynunet_block import get_conv_layer, OutBlock, ResBlock
from SSCFormer.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional as F

einops, _ = optional_import("einops")


class Conv3d_batchnorm(nn.Module):

    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride=(1, 1, 1), activation='relu'):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv3d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size,
                               stride=stride, padding=1) if kernel_size[0] == 3 else nn.Conv2d(
            in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size,
            stride=stride, padding=0)
        self.batchnorm = nn.BatchNorm3d(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        if self.activation == 'relu':
            return F.relu(x)
        else:
            return x


# state intra_scale resInception
class ISRI(nn.Module):

    def __init__(self, num_in_channels, num_filters, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.W = num_filters * alpha

        filt_cnt_3x3 = int(self.W * 0.167) + 1
        filt_cnt_5x5 = int(self.W * 0.333)
        filt_cnt_7x7 = int(self.W * 0.5)
        num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7

        self.shortcut = nn.Conv3d(in_channels=num_in_channels, out_channels=num_out_filters, kernel_size=(1, 1, 1))

        self.conv_3x3 = Conv3d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size=(3, 3, 3), activation='relu')

        self.conv_5x5 = Conv3d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size=(3, 3, 3), activation='relu')

        self.conv_7x7 = Conv3d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size=(3, 3, 3), activation='relu')

        self.batch_norm1 = nn.BatchNorm3d(num_out_filters)
        self.batch_norm2 = nn.BatchNorm3d(num_out_filters)

    def forward(self, x):
        shortcut = self.shortcut(x)

        a = self.conv_3x3(x)
        b = self.conv_5x5(a)
        c = self.conv_7x7(b)

        x = torch.cat([a, b, c], axis=1)
        x = self.batch_norm1(x)

        x = x + shortcut
        x = self.batch_norm2(x)
        x = F.relu(x)

        return x


# state spatial_channel transformer block
class SC_TransformerBlock(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        super().__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.sc = SC_aware(input_size=input_size, hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads,
                           channel_attn_drop=dropout_rate, spatial_attn_drop=dropout_rate)
        self.conv51 = ResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.sc(self.norm(x))

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv(attn)

        return x


# state spatial_channel_aware module
class SC_aware(nn.Module):

    def __init__(self, input_size, hidden_size, proj_size, num_heads=4, qkv_bias=False,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        self.E = self.F = nn.Linear(input_size, proj_size)

        self.attn_drop_1 = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj1 = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)

        qkvv = qkvv.permute(2, 0, 3, 1, 4)

        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]
        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        k_shared_projected = self.E(k_shared)

        v_SA_projected = self.F(v_SA)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature1

        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop_1(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C)

        # Concat fusion
        x_SA = self.out_proj1(x_SA)
        x_CA = self.out_proj2(x_CA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature1', 'temperature2'}


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.se = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


# state scaling attention
class SA(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


class Encoder(nn.Module):
    def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4], dims=[32, 64, 128, 256],
                 proj_size=[64, 64, 64, 32], depths=[3, 3, 3, 3], num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.0, transformer_dropout_rate=0.15):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        self.conv = nn.ModuleList()

        # define intra_scale resInception
        self.isri_layers = nn.ModuleList()
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(4, 4, 4), stride=(4, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )

        # define multi_scale pyramid
        self.scale_img1 = nn.AvgPool3d((4, 4, 4), (4, 4, 4))
        self.scale_img2 = nn.AvgPool3d((2, 2, 2), (2, 2, 2))

        self.downsample_layers.append(stem_layer)
        self.isri_layers.append(ISRI(dims[0], dims[0]))
        for i in range(3):
            self.conv.append(
                nn.Sequential(nn.Conv3d(in_channels, out_channels=dims[i], kernel_size=3, stride=1, padding=1)))
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i] * 2, dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)
            self.isri_layers.append(ISRI(dims[i + 1], dims[i + 1]))

        self.stages = nn.ModuleList()

        # define scaling attention
        self.sa_layers = nn.ModuleList()
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    SC_TransformerBlock(input_size=input_size[i], hidden_size=dims[i], proj_size=proj_size[i],
                                        num_heads=num_heads,
                                        dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.sa_layers.append(SA(channel=dims[i], reduction=16, kernel_size=7))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []
        mul_imgs = []

        scale_img_1 = self.scale_img1(x)
        mul_imgs.append(scale_img_1)
        scale_img_2 = self.scale_img2(scale_img_1)
        mul_imgs.append(scale_img_2)
        scale_img_3 = self.scale_img2(scale_img_2)
        mul_imgs.append(scale_img_3)

        x = self.downsample_layers[0](x)
        x = self.isri_layers[0](x)
        x = self.stages[0](x)
        x = self.sa_layers[0](x)
        hidden_states.append(x)

        for i in range(1, 4):
            x = torch.cat((F.relu(self.conv[i - 1](mul_imgs[i - 1])), x), axis=1)
            x = self.downsample_layers[i](x)
            x = self.isri_layers[i](x)
            x = self.stages[i](x)
            x = self.sa_layers[i](x)
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class UpBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.decoder_block = nn.ModuleList()
        if conv_decoder == True:
            self.decoder_block.append(
                ResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                         norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(
                    SC_TransformerBlock(input_size=out_size, hidden_size=out_channels, proj_size=proj_size,
                                        num_heads=num_heads,
                                        dropout_rate=0.15, pos_embed=True))
            stage_blocks.append(SA(channel=out_channels, reduction=16, kernel_size=7))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)
        return out


# Use the Attention of the Vision transformer to state the attention in the ISTB
class InterScaleAttention(nn.Module):
    def __init__(self, dim):
        super(InterScaleAttention, self).__init__()
        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.num_head = 8
        self.scale = (dim // self.num_head) ** 0.5

    def forward(self, x):
        B, num_blocks_d, num_blocks_h, num_blocks_w, _, C = x.shape  # (B, num_blocks, num_blocks, N, C)
        qkv = self.qkv_linear(x).reshape(B, num_blocks_d, num_blocks_h, num_blocks_w, -1, 3, self.num_head,
                                         C // self.num_head).permute(5, 0, 1, 2, 3, 6, 4, 7).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        atten = q @ k.transpose(-1, -2).contiguous()
        atten = self.softmax(atten)
        atten_value = (atten @ v).transpose(-2, -3).contiguous().reshape(B, num_blocks_d, num_blocks_h, num_blocks_w,
                                                                         -1, C)
        atten_value = self.proj(atten_value)
        return atten_value


# state inter_scale transformer bridge
class ISTB(nn.Module):
    def __init__(self, dim=256, num=1):
        super(ISTB, self).__init__()
        self.ini_win_size = 2
        self.channels = [32, 64, 128, 256]
        self.dim = dim
        self.depth = 4
        self.fc_module = nn.ModuleList()
        self.fc_rever_module = nn.ModuleList()
        self.num = num
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        for i in range(self.depth):
            self.fc_module.append(nn.Linear(self.channels[i], self.dim))

        for i in range(self.depth):
            self.fc_rever_module.append(nn.Linear(self.dim, self.channels[i]))

        self.scaleAtt = InterScaleAttention(dim)
        self.split_list = [8 * 8 * 8, 4 * 4 * 4, 2 * 2 * 2, 1 * 1 * 1]

        self.res_blocks = nn.ModuleList()
        for i in range(len(self.channels)):
            self.res_blocks.append(
                ResBlock(spatial_dims=3, in_channels=self.channels[i], out_channels=self.channels[i], kernel_size=3,
                         stride=1,
                         norm_name="batch"))

    def forward(self, x):
        x = [self.fc_module[i](item.permute(0, 2, 3, 4, 1)) for i, item in enumerate(x)]  # [(B, H, W, C)]
        for j, item in enumerate(x):
            B, D, H, W, C = item.shape
            win_size = self.ini_win_size ** (self.depth - j - 1)
            item = item.reshape(B, D // win_size, win_size, H // win_size, win_size, W // win_size, win_size,
                                C).permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
            item = item.reshape(B, D // win_size, H // win_size, W // win_size, win_size * win_size * win_size,
                                C).contiguous()
            x[j] = item

        x = tuple(x)
        x = torch.cat(x, dim=-2)

        x_pre = x
        x = self.norm(x)
        x = self.scaleAtt(x)
        x = x + x_pre
        x = torch.split(x, self.split_list, dim=-2)
        x = list(x)

        for j, item in enumerate(x):
            B, num_blocks_d, num_blocks_h, num_blocks_w, N, C = item.shape
            win_size = self.ini_win_size ** (self.depth - j - 1)
            item = item.reshape(B, num_blocks_d, num_blocks_h, num_blocks_w, win_size, win_size, win_size, C).permute(0,
                                                                                                                      1,
                                                                                                                      4,
                                                                                                                      2,
                                                                                                                      5,
                                                                                                                      3,
                                                                                                                      6,
                                                                                                                      7).contiguous().reshape(
                B,
                num_blocks_d * win_size,
                num_blocks_h * win_size,
                num_blocks_w * win_size,
                C)
            item = self.fc_rever_module[j](item).permute(0, 4, 1, 2, 3).contiguous()
            x[j] = item

        for i in range(len(self.channels)):
            x[i] = self.res_blocks[i](x[i])

        return x


class SSCFormer(SegmentationNetwork):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [64, 128, 128],
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (4, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size
        self.fusion_dim = [32, 64, 128, 256]
        self.num_module = 4

        # define encoder
        self.encoder = Encoder(dims=dims, depths=depths, num_heads=num_heads, in_channels=in_channels)

        # define fusion module
        self.istb = ISTB(dim=256)

        # define fusion module
        self.fusion_layers = nn.ModuleList()
        for i in range(self.num_module):
            self.fusion_layers.append(
                nn.Conv3d(self.fusion_dim[i] * 2, self.fusion_dim[i], 1, 1)
            )

        self.encoder1 = ResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 8 * 8,
        )
        self.decoder4 = UpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 16 * 16,
        )
        self.decoder3 = UpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 32 * 32,
        )
        self.decoder2 = UpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=64 * 128 * 128,
            conv_decoder=True,
        )
        self.out1 = OutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.up1 = nn.ConvTranspose3d(feature_size * 2, out_channels, (2, 2, 2), (2, 2, 2))
            self.up2 = nn.ConvTranspose3d(feature_size * 4, out_channels, (2, 2, 2), (2, 2, 2))

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, encoded_states = self.encoder(x_in)
        bridged_states = self.istb(encoded_states)

        # fusion module
        fusions = []
        for i in range(self.num_module):
            skip = self.fusion_layers[i](torch.cat((bridged_states[i], encoded_states[i]), dim=1))
            fusions.append(skip)

        convBlock = self.encoder1(x_in)

        # Four decoders
        dec4 = self.proj_feat(fusions[3], self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, fusions[2])
        dec2 = self.decoder4(dec3, fusions[1])
        dec1 = self.decoder3(dec2, fusions[0])

        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.up1(dec1), self.up2(dec2)]
        else:
            logits = self.out1(out)
        return logits
