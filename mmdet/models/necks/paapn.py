# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
from .fpn import FPN

import torch
# from torch import nn
import torch.fft as afft
from torch.autograd import Variable

import numpy as np
# 新增
# from scipy.fft import dct, idct
import random
from torch.nn import init
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..attention.ShuffleAttention import ShuffleAttention
from ..attention.CrissCrossAttention import CrissCrossAttention
from ..attention.S2Attention import S2Attention
from torch.nn.parameter import Parameter

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# def visualize_with_tsne(inputs, bilinear_inputs, layer_index):
#     # 选择要可视化的层
#     features_before = inputs[layer_index].view(inputs[layer_index].size(0), -1).cpu().detach().numpy()
#     features_after = bilinear_inputs[layer_index].view(bilinear_inputs[layer_index].size(0), -1).cpu().detach().numpy()
#
#     # 使用t-SNE进行降维
#     tsne = TSNE(n_components=2, random_state=0)
#     features_before_2d = tsne.fit_transform(features_before)
#     features_after_2d = tsne.fit_transform(features_after)
#
#     # 绘制可视化结果
#     plt.figure(figsize=(12, 6))
#
#     plt.subplot(121)
#     plt.scatter(features_before_2d[:, 0], features_before_2d[:, 1], marker='o')
#     plt.title('Features before HCBP')
#
#     plt.subplot(122)
#     plt.scatter(features_after_2d[:, 0], features_after_2d[:, 1], marker='o')
#     plt.title('Features after HCBP')
#
#     plt.show()
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

class DyNamicPyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""

    def __init__(self, channel_size=256, num_layers=None):
        super(DyNamicPyramidAttentions, self).__init__()

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
        return sparse_sketch_matrix.to_dense()  # 99.48%都是零元素
        # return sparse_sketch_matrix

class PyramidAttentions(nn.Module):
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

        # return [A1, A2, A3, A4, A5, A1_spatial, A2_spatial, A3_spatial, A4_spatial, A5_spatial]
        return [A1, A2, A3, A4, A5]


def gem(x, p=torch.ones(1) * 3, eps: float = 1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

class DynamicLearnPyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""

    def __init__(self, channel_size=256):
        super(DynamicLearnPyramidAttentions, self).__init__()

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

        # self.beta = Parameter(torch.ones(1)*3)#可以拿1作初始化，可以拿先验值做初始化

        self.alpha1 = nn.Parameter(torch.ones(1))
        self.beta1 = nn.Parameter(torch.ones(1))

        self.alpha2 = nn.Parameter(torch.ones(1))
        self.beta2 = nn.Parameter(torch.ones(1))

        self.alpha3 = nn.Parameter(torch.ones(1))
        self.beta3 = nn.Parameter(torch.ones(1))

        self.alpha4 = nn.Parameter(torch.ones(1))
        self.beta4 = nn.Parameter(torch.ones(1))

        self.alpha5 = nn.Parameter(torch.ones(1))
        self.beta5 = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        F1, F2, F3, F4, F5 = inputs#这里不处理第一层，处理后4层

        A1_spatial = self.A1_1(F1)
        A1_channel = self.A1_2(F1) # 原来写错成A2_1
        A1 = self.alpha1*A1_spatial * F1 + self.beta1*A1_channel * F1#Out[2]: torch.Size([2, 256, 84, 84])

        A2_spatial = self.A2_1(F2)
        A2_channel = self.A2_2(F2)
        A2_channel = (A2_channel + A1_channel) / 2#(原来写错成 A2_channel + A2_channel) / 2
        A2 = self.alpha2*A2_spatial * F2 + self.beta2*A2_channel * F2#Out[2]: torch.Size([2, 256, 42, 42])

        A3_spatial = self.A3_1(F3)
        A3_channel = self.A3_2(F3)
        A3_channel = (A3_channel + A2_channel) / 2
        A3 = self.alpha3*A3_spatial * F3 + self.beta3*A3_channel * F3#Out[3]: torch.Size([2, 256, 21, 21])

        A4_spatial = self.A4_1(F4)
        A4_channel = self.A4_2(F4)
        A4_channel = (A4_channel + A3_channel) / 2
        A4 = self.alpha4*A4_spatial * F4 + self.beta4*A4_channel * F4#Out[4]: torch.Size([2, 256, 11, 11])

        A5_spatial = self.A5_1(F5)
        A5_channel = self.A5_2(F5)
        A5_channel = (A5_channel + A4_channel) / 2
        A5 = self.alpha5*A5_spatial * F5 + self.beta5*A5_channel * F5#ut[5]: torch.Size([2, 256, 6, 6])

        return [A1, A2, A3, A4, A5]

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

class CAPyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""

    def __init__(self, channel_size=256):
        super(CAPyramidAttentions, self).__init__()

        # self.A1_1 = SpatialGate(channel_size)
        # self.A1_2 = ChannelGate(channel_size)
        self.A1_3 = CA(channel_size)

        # self.A2_1 = SpatialGate(channel_size)
        # self.A2_2 = ChannelGate(channel_size)
        self.A2_3 = CA(channel_size)

        # self.A3_1 = SpatialGate(channel_size)
        # self.A3_2 = ChannelGate(channel_size)
        self.A3_3 = CA(channel_size)

        # self.A4_1 = SpatialGate(channel_size)
        # self.A4_2 = ChannelGate(channel_size)
        self.A4_3 = CA(channel_size)

        # self.A5_1 = SpatialGate(channel_size)
        # self.A5_2 = ChannelGate(channel_size)
        self.A5_3 = CA(channel_size)
        # self.acmix = ACmix(in_planes=channel_size, out_planes=channel_size)

    def forward(self, inputs):
        F1, F2, F3, F4, F5 = inputs#这里不处理第一层，处理后4层

        # A1_spatial = self.A1_1(F1)
        # A1_channel = self.A1_1(F1)#A2_1
        A1_ca = self.A1_3(F1)  # A2_1

        # A1 = A1_spatial * F1 + A1_channel * F1 + A1_ca*F1#Out[2]: torch.Size([2, 256, 84, 84])
        A1 =  A1_ca * F1

        # A2_spatial = self.A2_1(F2)
        # A2_channel = self.A2_2(F2)
        A2_ca = self.A2_3(F2)

        # A2_channel = (A2_channel + A1_channel) / 2 # (A2_channel + A2_channel) / 2 写错了
        A2 = A2_ca*F2#Out[2]: torch.Size([2, 256, 42, 42])

        # A3_spatial = self.A3_1(F3)
        # A3_channel = self.A3_2(F3)
        A3_ca = self.A3_3(F3)

        # A3_channel = (A3_channel + A2_channel) / 2
        A3 = A3_ca*F3#Out[3]: torch.Size([2, 256, 21, 21])

        # A4_spatial = self.A4_1(F4)
        # A4_channel = self.A4_2(F4)
        A4_ca = self.A4_3(F4)

        # A4_channel = (A4_channel + A3_channel) / 2
        A4 = A4_ca*F4 #Out[4]: torch.Size([2, 256, 11, 11])

        # A5_spatial = self.A5_1(F5)
        # A5_channel = self.A5_2(F5)
        A5_ca = self.A5_3(F5)

        # A5_channel = (A5_channel + A4_channel) / 2
        A5 = A5_ca*F5#ut[5]: torch.Size([2, 256, 6, 6])

        return [A1, A2, A3, A4, A5]

class ShufflePyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""

    def __init__(self, channel_size=256):
        super(ShufflePyramidAttentions, self).__init__()

        self.A1_3 = ShuffleAttention(channel_size, reduction=16, G=8)
        self.A2_3 = ShuffleAttention(channel_size, reduction=16, G=8)
        self.A3_3 = ShuffleAttention(channel_size, reduction=16, G=8)
        self.A4_3 = ShuffleAttention(channel_size, reduction=16, G=8)
        self.A5_3 = ShuffleAttention(channel_size, reduction=16, G=8)

    def forward(self, inputs):
        F1, F2, F3, F4, F5 = inputs#这里不处理第一层，处理后4层

        A1_ca = self.A1_3(F1)  # A2_1
        A1 = A1_ca * F1

        A2_ca = self.A2_3(F2)
        A2 = A2_ca*F2#Out[2]: torch.Size([2, 256, 42, 42])

        A3_ca = self.A3_3(F3)
        A3 = A3_ca*F3#Out[3]: torch.Size([2, 256, 21, 21])

        A4_ca = self.A4_3(F4)
        A4 = A4_ca*F4 #Out[4]: torch.Size([2, 256, 11, 11])
        A5_ca = self.A5_3(F5)

        # A5_channel = (A5_channel + A4_channel) / 2
        A5 = A5_ca*F5#ut[5]: torch.Size([2, 256, 6, 6])

        return [A1, A2, A3, A4, A5]


class S2CAPyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""

    def __init__(self, channel_size=256):
        super(S2CAPyramidAttentions, self).__init__()

        self.A1_3 = CA(channel_size)
        self.A2_3 = CA(channel_size)
        self.A3_3 = CA(channel_size)
        self.A4_3 = CA(channel_size)
        self.A5_3 = CA(channel_size)

        self.A1_2 = S2Attention(channel_size)
        self.A2_2 = S2Attention(channel_size)
        self.A3_2 = S2Attention(channel_size)
        self.A4_2 = S2Attention(channel_size)
        self.A5_2 = S2Attention(channel_size)

        self.alpha1 = nn.Parameter(torch.ones(1))
        self.beta1 = nn.Parameter(torch.ones(1))

        self.alpha2 = nn.Parameter(torch.ones(1))
        self.beta2 = nn.Parameter(torch.ones(1))

        self.alpha3 = nn.Parameter(torch.ones(1))
        self.beta3 = nn.Parameter(torch.ones(1))

        self.alpha4 = nn.Parameter(torch.ones(1))
        self.beta4 = nn.Parameter(torch.ones(1))

        self.alpha5 = nn.Parameter(torch.ones(1))
        self.beta5 = nn.Parameter(torch.ones(1))

    def forward(self, inputs):
        F1, F2, F3, F4, F5 = inputs#这里不处理第一层，处理后4层

        A1_ca = self.A1_3(F1)  # A2_1
        A1_s2 = self.A1_2(F1)  # A2_1
        A1 = self.alpha1*A1_ca * F1 + self.beta1*A1_s2*F1

        A2_ca = self.A2_3(F2)
        A2_s2 = self.A2_2(F2)
        A2 = self.alpha2*A2_ca*F2 + self.beta2*A2_s2*F2

        A3_ca = self.A3_3(F3)
        A3_s2 = self.A3_2(F3)
        A3 = self.alpha3*A3_ca*F3 + self.beta3*A3_s2*F3

        A4_ca = self.A4_3(F4)
        A4_s2 = self.A4_2(F4)
        A4 = self.alpha4*A4_ca*F4 + self.beta4*A4_s2*F4

        A5_ca = self.A5_3(F5)
        A5_s2 = self.A5_2(F5)
        A5 = self.alpha5*A5_ca*F5 + self.beta5*A5_s2*F5#ut[5]: torch.Size([2, 256, 6, 6])

        return [A1, A2, A3, A4, A5]
class S2PyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""

    def __init__(self, channel_size=256):
        super(S2PyramidAttentions, self).__init__()

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

        self.A1_3 = S2Attention(channel_size)
        self.A2_3 = S2Attention(channel_size)
        self.A3_3 = S2Attention(channel_size)
        self.A4_3 = S2Attention(channel_size)
        self.A5_3 = S2Attention(channel_size)

        self.alpha1 = nn.Parameter(torch.ones(1))
        self.beta1 = nn.Parameter(torch.ones(1))
        self.gamma1 = nn.Parameter(torch.ones(1))

        self.alpha2 = nn.Parameter(torch.ones(1))
        self.beta2 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))

        self.alpha3 = nn.Parameter(torch.ones(1))
        self.beta3 = nn.Parameter(torch.ones(1))
        self.gamma3 = nn.Parameter(torch.ones(1))

        self.alpha4 = nn.Parameter(torch.ones(1))
        self.beta4 = nn.Parameter(torch.ones(1))
        self.gamma4 = nn.Parameter(torch.ones(1))

        self.alpha5 = nn.Parameter(torch.ones(1))
        self.beta5 = nn.Parameter(torch.ones(1))
        self.gamma5 = nn.Parameter(torch.ones(1))
        # self.acmix = ACmix(in_planes=channel_size, out_planes=channel_size)
    def forward(self, inputs):
        F1, F2, F3, F4, F5 = inputs#这里不处理第一层，处理后4层

        A1_spatial = self.A1_1(F1)
        A1_channel = self.A1_2(F1) # 原来写错成A2_1
        A1_s2 = self.A1_3(F1)  # A2_1
        A1 = self.alpha1*A1_spatial * F1 + self.beta1*A1_channel * F1 + self.gamma1*A1_s2*F1#Out[2]: torch.Size([2, 256, 84, 84])

        A2_spatial = self.A2_1(F2)
        A2_channel = self.A2_2(F2)
        A2_s2 = self.A2_3(F2)
        A2_channel = (A2_channel + A1_channel) / 2#(原来写错成 A2_channel + A2_channel) / 2
        A2 = self.alpha2*A2_spatial * F2 + self.beta2*A2_channel * F2 +self.gamma2*A2_s2*F2

        A3_spatial = self.A3_1(F3)
        A3_channel = self.A3_2(F3)
        A2_s3 = self.A3_3(F3)
        A3_channel = (A3_channel + A2_channel) / 2
        A3 = self.alpha3*A3_spatial * F3 + self.beta3*A3_channel * F3 +self.gamma3*A2_s3*F3

        A4_spatial = self.A4_1(F4)
        A4_channel = self.A4_2(F4)
        A2_s4 = self.A4_3(F4)
        A4_channel = (A4_channel + A3_channel) / 2
        A4 = self.alpha4*A4_spatial * F4 + self.beta4*A4_channel * F4 +self.gamma4*A2_s4*F4

        A5_spatial = self.A5_1(F5)
        A5_channel = self.A5_2(F5)
        A2_s5 = self.A5_3(F5)
        A5_channel = (A5_channel + A4_channel) / 2
        A5 = self.alpha5*A5_spatial * F5 + self.beta5*A5_channel * F5 +self.gamma5*A2_s5*F5

        return [A1, A2, A3, A4, A5]

@MODELS.register_module()
class PAAPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(PAAPN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            add_extra_convs,
            relu_before_extra_convs,
            no_norm_on_lateral,
            conv_cfg,
            norm_cfg,
            act_cfg,
            init_cfg=init_cfg)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):#明天在这里将下采样stride=2->stride=1 并且增加SPD模块
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

        self.compact_bilinear_pooling_layers = nn.ModuleList([
            CompactBilinearPooling(input_dim1=c, input_dim2=c, output_dim=c)  # 降低的维度和输入一致
            for c in self.in_channels
        ])
        # 构建注意金字塔
        self.apn = PyramidAttentions(channel_size=256)
        # self.dapn = DynamicLearnPyramidAttentions(channel_size=256)
        # self.CA = CAPyramidAttentions(channel_size=256)
        # self.shuffle = ShufflePyramidAttentions(channel_size=256)
        # self.crisscross = CrissCrossPyramidAttentions(channel_size=256)
        # self.S2CA = S2CAPyramidAttentions(channel_size=256)
        # self.S2Orin = S2PyramidAttentions(channel_size=256)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        mapped_outputs = []
        for i in range(len(inputs)):
            layer_input = inputs[i]  # torch.Size([2, 192, 84, 84])
            # 对每层进行紧凑双线性池化操作，包括了降维
            # bilinear_output = self.compact_bilinear_pooling_layers[i](layer_input)  # torch.Size([2, 2048])
            mapped_outputs.append(layer_input)
        # 将映射后的特征组合在一起
        bilinear_inputs = tuple(mapped_outputs)

        # 调用函数进行可视化
        # visualize_with_tsne(inputs, bilinear_inputs, layer_index=1)

        # inputs = tuple(mapped_outputs)
        # build laterals
        laterals = [
            lateral_conv(bilinear_inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        apn_out = self.apn(outs)
        # apn_out = self.dapn(outs)
        # apn_out = self.CA(outs)
        # apn_out = self.shuffle(outs)
        # apn_out = self.crisscross(outs)
        # apn_out = self.S2CA(outs)#(再试试这个)
        # apn_out = self.S2Orin(outs)
        return tuple(apn_out)
