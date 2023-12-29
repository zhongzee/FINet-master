# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmengine.model import BaseModule, constant_init, normal_init

from mmdet.registry import MODELS
from ..layers import DyReLU

import torch.fft as afft
from torch.autograd import Variable

import numpy as np
# 新增
import random
from torch.nn import init
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Reference:
# https://github.com/microsoft/DynamicHead
# https://github.com/jshilong/SEPC

class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.

    Args:

        output_dim: output dimension for compact bilinear pooling.

        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.

        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.

        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.

        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.

        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1

        self.sparse_sketch_matrix1 = Variable(self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim))

        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1

        self.sparse_sketch_matrix2 = Variable(self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim))

    def forward(self, bottom1, bottom2=None):
        """
        bottom1: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, input_dim2, height, width].
        """
        if bottom2 is None:
            bottom2 = bottom1.clone()

        assert bottom1.size(1) == self.input_dim1 and \
               bottom2.size(1) == self.input_dim2

        # load Variable `sparse_sketch_matrix1` to gpu # 注意这里必须要使用上GPU
        if self.sparse_sketch_matrix1.device != bottom1.device:
            self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1.to(bottom1.device)
            self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2.to(bottom1.device)

        batch_size, _, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)
        # # 替换为稀疏矩阵乘法
        # sketch_1 = torch.sparse.mm(self.sparse_sketch_matrix1, bottom1_flat.t()).t()
        # sketch_2 = torch.sparse.mm(self.sparse_sketch_matrix2, bottom2_flat.t()).t()
        #  使用RFFT替换FFT
        fft1 = afft.fft(sketch_1)#fft
        fft2 = afft.fft(sketch_2)
        fft_product = fft1 * fft2
        cbp_flat = afft.ifft(fft_product).real#ifft
        # 使用DCT替换FFT
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # dct1 = torch.from_numpy(dct(sketch_1.cpu().detach().numpy(), type=2, axis=-1, norm=None)).to(device)
        # dct2 = torch.from_numpy(dct(sketch_2.cpu().detach().numpy(), type=2, axis=-1, norm=None)).to(device)
        # dct_product = dct1 * dct2
        # dct_product_cpu = dct_product.cpu()
        # cbp_flat = torch.from_numpy(idct(dct_product_cpu.numpy(), type=2, axis=-1, norm=None)).to(device)

        cbp = cbp_flat.view(-1, height, width, self.output_dim)
        cbp = cbp.permute(0, 3, 1, 2).contiguous()  # 调整维度顺序以匹配输入形状

        # cbp = cbp_flat.view(batch_size, height, width, self.output_dim)
        # 这一行代码会沿着height和width维度（在此处分别是dim=1和dim=2）对cbp张量求和，从而移除这两个维度。最终，cbp张量的形状将变为[batch_size, output_dim]。
        # if self.sum_pool:
        #     cbp = cbp.sum(dim=1).sum(dim=1) # 不执行该行，保留空间信息有助于模型在空间上进行更精确的预测，做分类可以不保留

        cbp = torch.sign(cbp) * torch.sqrt(torch.abs(cbp) + 1e-10)
        cbp = torch.nn.functional.normalize(cbp)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert (rand_h.ndim == 1 and rand_s.ndim ==
                1 and len(rand_h) == len(rand_s))
        assert (np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()  # 99.48%都是零元

class SpatialGate(nn.Module):
    """generation spatial attention mask"""

    def __init__(self, out_channels):
        super(SpatialGate, self).__init__()
        self.conv = nn.ConvTranspose2d(out_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(x)


class ChannelGate(nn.Module):
    """generation channel attention mask"""

    def __init__(self, out_channels):
        super(ChannelGate, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels // 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels // 16, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = nn.AdaptiveAvgPool2d(output_size=1)(x)
        x = F.relu(self.conv1(x), inplace=True)
        x = torch.sigmoid(self.conv2(x))

class PyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""

    def __init__(self, channel_size=256):
        super(PyramidAttentions, self).__init__()

        self.A1_1 = SpatialGate(channel_size)
        self.A1_2 = ChannelGate(channel_size)

        self.A2_1 = SpatialGate(channel_size)
        self.A2_2 = ChannelGate(channel_size)

        self.A3_1 = SpatialGate(channel_size)
        self.A3_2 = ChannelGate(channel_size)

        self.A4_1 = SpatialGate(channel_size)
        self.A4_2 = ChannelGate(channel_size)

        self.A5_1 = SpatialGate(channel_size)
        self.A5_2 = ChannelGate(channel_size)
        # self.acmix = ACmix(in_planes=channel_size, out_planes=channel_size)
    def forward(self, inputs):
        F1, F2, F3, F4, F5 = inputs#这里不处理第一层，处理后4层

        A1_spatial = self.A1_1(F1)
        A1_channel = self.A1_2(F1) # 原来写错成A2_1
        A1 = A1_spatial * F1 + A1_channel * F1#Out[2]: torch.Size([2, 256, 84, 84])

        A2_spatial = self.A2_1(F2)
        A2_channel = self.A2_2(F2)
        A2_channel = (A2_channel + A1_channel) / 2#(原来写错成 A2_channel + A2_channel) / 2
        A2 = A2_spatial * F2 + A2_channel * F2#Out[2]: torch.Size([2, 256, 42, 42])

        A3_spatial = self.A3_1(F3)
        A3_channel = self.A3_2(F3)
        A3_channel = (A3_channel + A2_channel) / 2
        A3 = A3_spatial * F3 + A3_channel * F3#Out[3]: torch.Size([2, 256, 21, 21])

        A4_spatial = self.A4_1(F4)
        A4_channel = self.A4_2(F4)
        A4_channel = (A4_channel + A3_channel) / 2
        A4 = A4_spatial * F4 + A4_channel * F4#Out[4]: torch.Size([2, 256, 11, 11])

        A5_spatial = self.A5_1(F5)
        A5_channel = self.A5_2(F5)
        A5_channel = (A5_channel + A4_channel) / 2
        A5 = A5_spatial * F5 + A5_channel * F5#ut[5]: torch.Size([2, 256, 6, 6])

        return [A1, A2, A3, A4, A5, A1_spatial, A2_spatial, A3_spatial, A4_spatial, A5_spatial]


class DyDCNv2(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.

    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset, mask)
        if self.with_norm:
            x = self.norm(x)
        return x


class DyHeadBlock(nn.Module):
    """DyHead Block with three types of attention.

    HSigmoid arguments in default act_cfg follow official code, not paper.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        zero_init_offset (bool, optional): Whether to use zero init for
            `spatial_conv_offset`. Default: True.
        act_cfg (dict, optional): Config dict for the last activation layer of
            scale-aware attention. Default: dict(type='HSigmoid', bias=3.0,
            divisor=6.0).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 zero_init_offset=True,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        self.zero_init_offset = zero_init_offset
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3

        self.spatial_conv_high = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_mid = DyDCNv2(in_channels, out_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, out_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(out_channels, 1, 1),
            nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.task_attn_module = DyReLU(out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)

    def forward(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            # calculate offset and mask of DCNv2 from middle-level feature
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(x[level], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], offset, mask),
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat / summed_levels))

        return outs


@MODELS.register_module()
class DyHead(BaseModule):
    """DyHead neck consisting of multiple DyHead Blocks.

    See `Dynamic Head: Unifying Object Detection Heads with Attentions
    <https://arxiv.org/abs/2106.08322>`_ for details.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_blocks (int, optional): Number of DyHead Blocks. Default: 6.
        zero_init_offset (bool, optional): Whether to use zero init for
            `spatial_conv_offset`. Default: True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=6,
                 zero_init_offset=True,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.zero_init_offset = zero_init_offset

        dyhead_blocks = []
        for i in range(num_blocks):
            in_channels = self.in_channels if i == 0 else self.out_channels
            dyhead_blocks.append(
                DyHeadBlock(
                    in_channels,
                    self.out_channels,
                    zero_init_offset=zero_init_offset))
        self.dyhead_blocks = nn.Sequential(*dyhead_blocks)

        # self.compact_bilinear_pooling_layers = nn.ModuleList([
        #     CompactBilinearPooling(input_dim1=c, input_dim2=c, output_dim=c)  # 降低的维度和输入一致
        #     for c in self.in_channels
        # ])
        # 构建注意金字塔
        # self.apn = PyramidAttentions(channel_size=256)

    def forward(self, inputs):
        """Forward function."""
        assert isinstance(inputs, (tuple, list))

        # 将映射后的特征组合在一起
        outs = self.dyhead_blocks(inputs)
        return tuple(outs)
