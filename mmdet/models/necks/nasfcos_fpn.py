# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops.merge_cells import ConcatCell
from mmengine.model import BaseModule, caffe2_xavier_init

from mmdet.registry import MODELS

# 新增
import torch
# from torch import nn
import torch.fft as afft
from torch.autograd import Variable
import numpy as np
import random
from torch.nn import init
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CA(nn.Module):
    def __init__(self, inp, reduction=32):#self, inp, oup, reduction=32
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out

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
        # 替换为稀疏矩阵乘法
        # sketch_1 = torch.sparse.mm(self.sparse_sketch_matrix1, bottom1_flat.t()).t()
        # sketch_2 = torch.sparse.mm(self.sparse_sketch_matrix2, bottom2_flat.t()).t()

        fft1 = afft.fft(sketch_1)
        fft2 = afft.fft(sketch_2)

        fft_product = fft1 * fft2

        cbp_flat = afft.ifft(fft_product).real

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
        return sparse_sketch_matrix.to_dense()  # 99.48%都是零元素
        # return sparse_sketch_matrix

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
        A1_channel = self.A2_1(F1) # A2_1
        A1 = A1_spatial * F1 + A1_channel * F1#Out[2]: torch.Size([2, 256, 84, 84])

        A2_spatial = self.A2_1(F2)
        A2_channel = self.A2_2(F2)
        A2_channel = (A2_channel + A2_channel) / 2#(A2_channel + A2_channel) / 2
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

class DynamicPyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""

    def __init__(self, channel_size=256, num_layers=4):
        super(DynamicPyramidAttentions, self).__init__()

        self.num_layers = num_layers
        self.spatial_gates = nn.ModuleList([SpatialGate(channel_size) for _ in range(num_layers)])
        self.channel_gates = nn.ModuleList([ChannelGate(channel_size) for _ in range(num_layers)])

    def forward(self, inputs):
        assert len(inputs) == self.num_layers, f"Input size mismatch: {len(inputs)} vs {self.num_layers}"

        output = []
        spatial_outputs = []
        prev_channel_attention = None

        for i in range(self.num_layers):
            spatial_attention = self.spatial_gates[i](inputs[i])
            channel_attention = self.channel_gates[i](inputs[i])

            if prev_channel_attention is not None:
                channel_attention = (channel_attention + prev_channel_attention) / 2

            feature_attention = spatial_attention * inputs[i] + channel_attention * inputs[i]
            output.append(feature_attention)
            spatial_outputs.append(spatial_attention)

            prev_channel_attention = channel_attention

        return output + spatial_outputs



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
        return x

@MODELS.register_module()
class NASFCOS_FPN(BaseModule):
    """FPN structure in NASFPN.

    Implementation of paper `NAS-FCOS: Fast Neural Architecture Search for
    Object Detection <https://arxiv.org/abs/1906.04423>`_

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): It decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=1,
                 end_level=-1,
                 add_extra_convs=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(NASFCOS_FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs

        self.adapt_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            adapt_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                stride=1,
                padding=0,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU', inplace=False))
            self.adapt_convs.append(adapt_conv)

        # C2 is omitted according to the paper
        extra_levels = num_outs - self.backbone_end_level + self.start_level

        def build_concat_cell(with_input1_conv, with_input2_conv):
            cell_conv_cfg = dict(
                kernel_size=1, padding=0, bias=False, groups=out_channels)
            return ConcatCell(
                in_channels=out_channels,
                out_channels=out_channels,
                with_out_conv=True,
                out_conv_cfg=cell_conv_cfg,
                out_norm_cfg=dict(type='BN'),
                out_conv_order=('norm', 'act', 'conv'),
                with_input1_conv=with_input1_conv,
                with_input2_conv=with_input2_conv,
                input_conv_cfg=conv_cfg,
                input_norm_cfg=norm_cfg,
                upsample_mode='nearest')

        # Denote c3=f0, c4=f1, c5=f2 for convince
        self.fpn = nn.ModuleDict()
        self.fpn['c22_1'] = build_concat_cell(True, True)
        self.fpn['c22_2'] = build_concat_cell(True, True)
        self.fpn['c32'] = build_concat_cell(True, False)
        self.fpn['c02'] = build_concat_cell(True, False)
        self.fpn['c42'] = build_concat_cell(True, True)
        self.fpn['c36'] = build_concat_cell(True, True)
        self.fpn['c61'] = build_concat_cell(True, True)  # f9
        self.extra_downsamples = nn.ModuleList()
        for i in range(extra_levels):
            extra_act_cfg = None if i == 0 \
                else dict(type='ReLU', inplace=False)
            self.extra_downsamples.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    act_cfg=extra_act_cfg,
                    order=('act', 'norm', 'conv')))

        self.compact_bilinear_pooling_layers = nn.ModuleList([
            CompactBilinearPooling(input_dim1=c, input_dim2=c, output_dim=c)  # 降低的维度和输入一致
            for c in self.in_channels
        ])
        # 构建注意金字塔
        self.apn = PyramidAttentions(channel_size=256)
        self.dapn = DynamicPyramidAttentions(channel_size=256, num_layers=5)

    def forward(self, inputs):
        """Forward function."""
        # 对每一层进行双线性池化操作，并进行维度映射，然后特征融合
        mapped_outputs = []
        for i in range(len(inputs)):
            layer_input = inputs[i]  # torch.Size([2, 192, 84, 84])
            # 对每层进行紧凑双线性池化操作，包括了降维
            bilinear_output = self.compact_bilinear_pooling_layers[i](layer_input)  # torch.Size([2, 2048])
            mapped_outputs.append(bilinear_output)
        # 将映射后的特征组合在一起
        inputs = tuple(mapped_outputs)

        feats = [
            adapt_conv(inputs[i + self.start_level])
            for i, adapt_conv in enumerate(self.adapt_convs)
        ]

        for (i, module_name) in enumerate(self.fpn):
            idx_1, idx_2 = int(module_name[1]), int(module_name[2])
            res = self.fpn[module_name](feats[idx_1], feats[idx_2])
            feats.append(res)

        ret = []
        for (idx, input_idx) in zip([9, 8, 7], [0, 1, 2]):  #【1，2，3】 add P2, P3, P4, P5 [9, 8, 7, 6], [0, 1, 2, 3]
            feats1, feats2 = feats[idx], feats[5]
            feats2_resize = F.interpolate(
                feats2,
                size=feats1.size()[2:],
                mode='bilinear',
                align_corners=False)

            feats_sum = feats1 + feats2_resize
            ret.append(
                F.interpolate(
                    feats_sum,
                    size=inputs[input_idx].size()[2:],
                    mode='bilinear',
                    align_corners=False))

        for submodule in self.extra_downsamples:
            ret.append(submodule(ret[-1]))

        ### 这里直接对FPN的每一层输出构建注意金字塔，并将FPN的输出与APN的输出相加
        # apn_out = self.dapn(ret)
        apn_out = self.apn(ret)  # f3_att, f4_att, f5_att, f6_att, a3, a4, a5, a6;只有空间

        return tuple(apn_out)

    def init_weights(self):
        """Initialize the weights of module."""
        super(NASFCOS_FPN, self).init_weights()
        for module in self.fpn.values():
            if hasattr(module, 'conv_out'):
                caffe2_xavier_init(module.out_conv.conv)

        for modules in [
                self.adapt_convs.modules(),
                self.extra_downsamples.modules()
        ]:
            for module in modules:
                if isinstance(module, nn.Conv2d):
                    caffe2_xavier_init(module)
