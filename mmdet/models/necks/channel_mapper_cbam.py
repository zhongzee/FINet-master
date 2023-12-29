# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS
##############
# detr neck层添加 CBAM 注意力机制
import numpy as np
import torch
from torch import nn
from torch.nn import init

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
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
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual
##############


@MODELS.register_module()
class ChannelMapper(BaseModule):
    r"""Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU').
        num_outs (int, optional): Number of output feature maps. There
            would be extra_convs when num_outs larger than the length
            of in_channels.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = ChannelMapper(in_channels, 11, 3).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 num_outs=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(ChannelMapper, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    ConvModule(
                        in_channel,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

        #初始化CBAM
        self.cbam =CBAMBlock(channel=64)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.convs)
        # 添加cbam注意力机制 这里的inputs是输出resnet三个层，不同大小的输出
        # outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        # 给backbone的每层输出 都加一个双线性注意力机制
        inputs_cbam = [self.cbam[i](inputs[i]) for i in range(len(inputs))]
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs_cbam))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)
