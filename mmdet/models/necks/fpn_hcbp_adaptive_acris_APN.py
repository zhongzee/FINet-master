# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType

import torch
# from torch import nn
# from scipy.fft import dct, idct
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
from torch.nn import Softmax

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

        fft1 = afft.fft(sketch_1)#rfft（下次试试实数FFT）
        fft2 = afft.fft(sketch_2)
        fft_product = fft1 * fft2
        cbp_flat = afft.ifft(fft_product).real#ifft
        # cbp_flat = torch.fft.irfft(fft_product)
        #
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
        return sparse_sketch_matrix.to_dense()  # 99.48%都是零元素
        # return sparse_sketch_matrix


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)

        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)

        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width).to(device)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

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
        x = F.relu(self.conv1(x), inplace=False)
        x = torch.sigmoid(self.conv2(x))
        return x

# class PyramidAttentions(nn.Module):
#     """Attention pyramid module with bottom-up attention pathway"""
#
#     def __init__(self, channel_size=256):
#         super(PyramidAttentions, self).__init__()
#
#         self.A1_1 = SpatialGate(channel_size)
#         self.A1_2 = ChannelGate(channel_size)
#
#         self.A2_1 = SpatialGate(channel_size)
#         self.A2_2 = ChannelGate(channel_size)
#
#         self.A3_1 = CrissCrossAttention(channel_size)
#         self.A3_2 = ChannelGate(channel_size)
#
#         self.A4_1 = CrissCrossAttention(channel_size)
#         self.A4_2 = ChannelGate(channel_size)
#
#         self.A5_1 = CrissCrossAttention(channel_size)
#         self.A5_2 = ChannelGate(channel_size)
#
#     def forward(self, inputs):
#         F1, F2, F3, F4, F5 = inputs
#
#         A1_spatial = self.A1_1(F1)
#         A1_channel = self.A1_2(F1)
#         A1 = A1_spatial * F1 + A1_channel * F1
#
#         A2_spatial = self.A2_1(F2)
#         A2_channel = self.A2_2(F2)
#         A2_channel = (A2_channel + A1_channel) / 2
#         A2 = A2_spatial * F2 + A2_channel * F2
#
#         A3_spatial = self.A3_1(F3)
#         A3_channel = self.A3_2(F3)
#         A3_channel = (A3_channel + A2_channel) / 2
#         A3 = A3_spatial * F3 + A3_channel * F3
#
#         A4_spatial = self.A4_1(F4)
#         A4_channel = self.A4_2(F4)
#         A4_channel = (A4_channel + A3_channel) / 2
#         A4 = A4_spatial * F4 + A4_channel * F4
#
#         A5_spatial = self.A5_1(F5)
#         A5_channel = self.A5_2(F5)
#         A5_channel = (A5_channel + A4_channel) / 2
#         A5 = A5_spatial * F5 + A5_channel * F5
#
#         return [A1, A2, A3, A4, A5]

class PyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""

    def __init__(self, num_layers=5, channel_size=256):
        super(PyramidAttentions, self).__init__()

        self.num_layers = num_layers
        self.attentions = nn.ModuleList()

        for i in range(num_layers):
            if i < num_layers // 2 or (i == num_layers // 2 and num_layers % 2 == 0):
                self.attentions.append(nn.Sequential(
                    SpatialGate(channel_size),
                    ChannelGate(channel_size)
                ))
            else:
                self.attentions.append(nn.Sequential(
                    CrissCrossAttention(channel_size),
                    ChannelGate(channel_size)
                ))

    def forward(self, inputs):
        assert len(inputs) == self.num_layers

        features = []
        prev_channel = 0
        for i in range(self.num_layers):
            spatial_out = self.attentions[i][0](inputs[i])
            channel_out = self.attentions[i][1](inputs[i])
            if i > 0:
                channel_out = (channel_out + prev_channel) / 2
            feature = spatial_out * inputs[i] + channel_out * inputs[i]
            features.append(feature)
            prev_channel = channel_out

        return features


@MODELS.register_module()
class FPN_HCBP_ACRIS_APN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])

        torch.Size([2, 192, 92, 92])
        torch.Size([2, 384, 46, 46])
        torch.Size([2, 768, 23, 23])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        self.compact_bilinear_pooling_layers = nn.ModuleList([
            CompactBilinearPooling(input_dim1=c, input_dim2=c, output_dim=c)#降低的维度和输入一致
            for c in self.in_channels
        ])
        # 构建自适应交替注意金字塔
        self.apn = PyramidAttentions(num_layers=self.num_outs, channel_size=256)
        # self.CA = CAPyramidAttentions(channel_size=256)
        # self.ca = CA(inp=256)
        # 在__init__函数中定义一个self.conv1x1的卷积层列表
        # self.conv1x1 = nn.ModuleList([
        #     nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        #     for _ in range(5)
        # ])
        #
        # # 在__init__函数中定义一个共享的1x1卷积层
        # self.shared_conv1x1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)

        # self.ca = nn.ModuleList([
        #     CA(inp=c)#降低的维度和输入一致
        #     for c in self.in_channels
        # ])
        # self.acmix = nn.ModuleList([
        #     ACmix(in_planes=c, out_planes=c)#降低的维度和输入一致
        #     for c in self.in_channels
        # ])
        # FPN之前就用这个
        # self.a2a = nn.ModuleList([
        #     DoubleAttention(in_channels=c, c_m=128, c_n=128, reconstruct=True)#降低的维度和输入一致
        #     for c in self.in_channels
        # ])

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)
        # 对每一层进行双线性池化操作，并进行维度映射，然后特征融合
        mapped_outputs = []
        for i in range(len(inputs)):
            layer_input = inputs[i]  # torch.Size([2, 192, 84, 84])
            # 对每层进行紧凑双线性池化操作，包括了降维
            bilinear_output = self.compact_bilinear_pooling_layers[i](layer_input)  # torch.Size([2, 2048])
            mapped_outputs.append(bilinear_output)
            # mapped_outputs.append(bilinear_output)
        # 将映射后的特征组合在一起
        inputs = tuple(mapped_outputs)
        # 这里再对每一层输出构建acmix注意金字塔
        # 这里再对每一层输出构建acmix注意金字塔
        # num_inputs = len(inputs)
        # acmix_outs = []
        # for i in range(num_inputs):
        #     layer_input = inputs[i]
        #     # in_planes = inputs[i].shape[1]  # 获取输入通道数
        #     acmix = self.a2a[i](layer_input)  # 初始化ACmix
        #
        #     # acmix_output = acmix(inputs[i])  # 计算输出
        #     acmix_outs.append(acmix)
        # acmix_out = tuple(acmix_outs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        ### 这里直接对FPN的每一层输出构建acmix注意金字塔，并将FPN的输出与APN的输出相加
        apn_out = self.apn(outs)  # f3_att, f4_att, f5_att, f6_att, a3, a4, a5, a6;只有空间 #没有APN
        # f1_att, f2_att, f3_att, f4_att, f5_att, a1, a2, a3, a4, a5 = apn_out
        # img_h = 736
        # img_w = 736
        # # # ROI operations
        # # # roi pyramid # 这里对X2的后一层做ROI特征金字塔
        # roi_1 = self.get_att_roi(a1, 2 ** 1, 16, img_h, img_w, iou_thred=0.05, topk=9)#torch.Size([36, 6])#RuntimeError: stack expects a non-empty TensorList
        # roi_2 = self.get_att_roi(a2, 2 ** 2, 32, img_h, img_w, iou_thred=0.05, topk=7)#torch.Size([28, 6])
        # roi_3 = self.get_att_roi(a3, 2 ** 3, 64, img_h, img_w, iou_thred=0.05, topk=5)#torch.Size([10, 6])
        # roi_4 = self.get_att_roi(a4, 2 ** 4, 128, img_h, img_w, iou_thred=0.05, topk=3)#torch.Size([4, 6])
        # roi_5 = self.get_att_roi(a5, 2 ** 5, 256, img_h, img_w, iou_thred=0.05, topk=1)#torch.Size([2, 6])
        # roi_list = [roi_1, roi_2, roi_3, roi_4, roi_5] # roi_list包含了不同尺度的ROI信息
        # roi_list = [roi_3, roi_4, roi_5]  # roi_list包含了不同尺度的ROI信息
        # x2.size():torch.Size([4, 192, 88, 88])
        # # stage II
        # x2_crop_resize, _ = self.get_roi_crop_N_feat(f1_att, roi_list, 2 ** 1)#torch.Size([4, 192, 88, 88])
        # x3_crop_resize, _ = self.get_roi_crop_N_feat(f2_att, roi_list, 2 ** 2)#AttributeError: 'tuple' object has no attribute 'size'
        # x4_crop_resize, _ = self.get_roi_crop_N_feat(f3_att, roi_list, 2 ** 3)#x3_crop_resize不对
        # x5_crop_resize, _ = self.get_roi_crop_N_feat(f4_att, roi_list, 2 ** 4)
        # x6_crop_resize, _ = self.get_roi_crop_N_feat(f5_att, roi_list, 2 ** 5)

        # x2_crop_resize, _ = self.get_roi_crop_N_feat(x2, roi_list, 2 ** 1)#torch.Size([4, 192, 88, 88])
        # x3_crop_resize = self.get_roi_crop_N_feat(x2_crop_resize, roi_list, 2 ** 2)#AttributeError: 'tuple' object has no attribute 'size'
        # x4_crop_resize = self.get_roi_crop_N_feat(x3_crop_resize[0], roi_list, 2 ** 3)#x3_crop_resize不对
        # x5_crop_resize = self.get_roi_crop_N_feat(x4_crop_resize[0], roi_list, 2 ** 4)
        # x6_crop_resize = self.get_roi_crop_N_feat(x5_crop_resize[0], roi_list, 2 ** 5)
        # return acmix_outs #把backbone输出的特征和fpn结构都传回来，后续有用
        return tuple(apn_out)
