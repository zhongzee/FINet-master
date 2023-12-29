# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import deform_conv2d
from mmengine import MessageHub
from mmengine.config import ConfigDict
from mmengine.model import bias_init_with_prob, normal_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..task_modules.prior_generators import anchor_inside_flags
from ..utils import (filter_scores_and_topk, images_to_levels, multi_apply,
                     sigmoid_geometric_mean, unmap)
from .atss_head import ATSSHead
import numpy as np
import random

class PyramidAttentions(nn.Module):
    """Attention pyramid module with bottom-up attention pathway"""

    def __init__(self, channel_size=256, num_inputs=3):
        super(PyramidAttentions, self).__init__()
        self.num_inputs = num_inputs

        self.attentions = nn.ModuleList([
            nn.ModuleList([
                SpatialGate(channel_size),
                ChannelGate(channel_size),
            ]) for _ in range(num_inputs)
        ])

    def forward(self, inputs):
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, but got {len(inputs)}")

        attentions = []
        for i, input_ in enumerate(inputs):
            F = input_
            A_spatial = self.attentions[i][0](F)
            A_channel = self.attentions[i][1](F)
            A = A_spatial * F + A_channel * F
            attentions.append(A)

            if i > 0:
                A_channel = (A_channel + attentions[i - 1].mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)) / 2
                A = A_spatial * F + A_channel * F
                attentions[i] = A

        return attentions


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


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

def generate_anchors_single_pyramid(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels.
    anchor_stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    box_centers = np.stack(
        [box_centers_x, box_centers_y], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_widths, box_heights], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (x1, y1, x2, y2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return torch.from_numpy(boxes)

def pth_nms(P: torch.tensor, thresh_iou: float):
    """Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """

    # we extract coordinates for every
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]

    # we extract the confidence scores as well
    scores = P[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:

        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(P[idx])

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]

        # find the IoU of every prediction in P with S
        IoU = inter / union

        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]

    return torch.stack(keep, dim=0)

class MyLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(MyLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
            Defaults to 8.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional):  Config dict for
        normalization layer. Defaults to None.
    """

    def __init__(self,
                 feat_channels: int,
                 stacked_convs: int,
                 la_down_rate: int = 8,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None) -> None:
        super().__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.stacked_convs,
                1,
                padding=0), nn.Sigmoid())

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=norm_cfg is None)

    def init_weights(self) -> None:
        """Initialize the parameters."""
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self,
                feat: Tensor,
                avg_feat: Optional[Tensor] = None) -> Tensor:
        """Forward function of task decomposition module."""
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))#x_inter
        # 通过两个1x1卷积层和一个Sigmoid激活函数，将avg_feat转换成一个与feat相同大小的权重矩阵weight，用来对feat进行加权。
        weight = self.layer_attention(avg_feat)# torch.Size([2, 6, 1, 1]) 论文提出了一种层注意机制(layer attention mechanism)，通过在层级动态计算任务特征来鼓励任务分解,也就是w

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs. self.reduction_conv.conv就是交互特征堆栈
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)#torch.Size([2, 256, 6, 256])
        conv_weight = conv_weight.reshape(b, self.feat_channels, # wk
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w) # xinter_k
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,w)#这里的feat就是X_k,inter,conv_weight就是wk
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat) #

        return feat


@MODELS.register_module()
class TOODHead(ATSSHead):
    """TOODHead used in `TOOD: Task-aligned One-stage Object Detection.

    <https://arxiv.org/abs/2108.07755>`_.

    TOOD uses Task-aligned head (T-head) and is optimized by Task Alignment
    Learning (TAL).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        num_dcn (int): Number of deformable convolution in the head.
            Defaults to 0.
        anchor_type (str): If set to ``anchor_free``, the head will use centers
            to regress bboxes. If set to ``anchor_based``, the head will
            regress bboxes based on anchors. Defaults to ``anchor_free``.
        initial_loss_cls (:obj:`ConfigDict` or dict): Config of initial loss.

    Example:
        >>> self = TOODHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 num_dcn: int = 0,
                 anchor_type: str = 'anchor_free',
                 initial_loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     activated=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 **kwargs) -> None:
        assert anchor_type in ['anchor_free', 'anchor_based']
        self.num_dcn = num_dcn
        self.anchor_type = anchor_type
        super().__init__(
            num_classes=num_classes, in_channels=in_channels, **kwargs)

        if self.train_cfg:
            self.initial_epoch = self.train_cfg['initial_epoch']
            self.initial_assigner = TASK_UTILS.build(
                self.train_cfg['initial_assigner'])
            self.initial_loss_cls = MODELS.build(initial_loss_cls)
            self.assigner = self.initial_assigner
            self.alignment_assigner = TASK_UTILS.build(
                self.train_cfg['assigner'])
            self.alpha = self.train_cfg['alpha']
            self.beta = self.train_cfg['beta']

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.inter_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if i < self.num_dcn:#self.num_dcn默认为2，也就是特征交互堆栈，第一层是可变形卷积，后面都是普通卷积，这个可以改
                conv_cfg = dict(type='DCNv2', deform_groups=4)
            else:
                conv_cfg = self.conv_cfg
            chn = self.in_channels if i == 0 else self.feat_channels
            self.inter_convs.append(#任务交互特征堆栈
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.cls_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            self.conv_cfg, self.norm_cfg)
        self.reg_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            self.conv_cfg, self.norm_cfg)

        self.tood_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.tood_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)

        self.cls_prob_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs,
                      self.feat_channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, 3, padding=1))
        self.reg_offset_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs,
                      self.feat_channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 4 * 2, 3, padding=1))

        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

        self.apn = PyramidAttentions(channel_size=256, num_inputs=5)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for m in self.inter_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_prob_module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_offset_module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.cls_prob_module[-1], std=0.01, bias=bias_cls)

        self.cls_decomp.init_weights()
        self.reg_decomp.init_weights()

        normal_init(self.tood_cls, std=0.01, bias=bias_cls)
        normal_init(self.tood_reg, std=0.01)

    def get_roi_crop_feat(self, x, roi_list, scale):
        """ROI guided refinement: ROI guided Zoom-in & ROI guided Dropblock"""
        n, c, x2_h, x2_w = x.size()
        roi_3, roi_4, roi_5 = roi_list
        roi_all = torch.cat([roi_3, roi_4, roi_5], 0)
        x2_ret = []
        crop_info_all = []
        if self.training:
            for i in range(n):
                roi_all_i = roi_all[roi_all[:, 0] == i] / scale
                xx1_resize, yy1_resize, = torch.min(roi_all_i[:, 1:3], 0)[0]
                xx2_resize, yy2_resize = torch.max(roi_all_i[:, 3:5], 0)[0]
                roi_3_i = roi_3[roi_3[:, 0] == i] / scale
                roi_4_i = roi_4[roi_4[:, 0] == i] / scale
                # alway drop the roi with highest score
                mask_un = torch.ones(c, x2_h, x2_w).to(x.device)
                pro_rand = random.random()
                if pro_rand < 0.3:
                    ind_rand = random.randint(0, roi_3_i.size(0) - 1)
                    xx1_drop, yy1_drop = roi_3_i[ind_rand, 1:3]
                    xx2_drop, yy2_drop = roi_3_i[ind_rand, 3:5]
                    mask_un[:, yy1_drop.long():yy2_drop.long(), xx1_drop.long():xx2_drop.long()] = 0
                elif pro_rand < 0.6:
                    ind_rand = random.randint(0, roi_4_i.size(0) - 1)
                    xx1_drop, yy1_drop = roi_4_i[ind_rand, 1:3]
                    xx2_drop, yy2_drop = roi_4_i[ind_rand, 3:5]
                    mask_un[:, yy1_drop.long():yy2_drop.long(), xx1_drop.long():xx2_drop.long()] = 0
                x2_drop = x[i] * mask_un
                x2_crop = x2_drop[:, yy1_resize.long():yy2_resize.long(),
                          xx1_resize.long():xx2_resize.long()].contiguous().unsqueeze(0)
                # normalize
                scale_rate = c * (yy2_resize - yy1_resize) * (xx2_resize - xx1_resize) / torch.sum(
                    mask_un[:, yy1_resize.long():yy2_resize.long(),
                    xx1_resize.long():xx2_resize.long()])
                x2_crop = x2_crop * scale_rate

                x2_crop_resize = F.interpolate(x2_crop, (x2_h, x2_w), mode='bilinear', align_corners=False)
                x2_ret.append(x2_crop_resize)

                crop_info = [xx1_resize, xx2_resize, yy1_resize, yy2_resize]
                crop_info_all.append(crop_info)
        else:
            for i in range(n):
                roi_all_i = roi_all[roi_all[:, 0] == i] / scale
                xx1_resize, yy1_resize, = torch.min(roi_all_i[:, 1:3], 0)[0]
                xx2_resize, yy2_resize = torch.max(roi_all_i[:, 3:5], 0)[0]
                x2_crop = x[i, :, yy1_resize.long():yy2_resize.long(),
                          xx1_resize.long():xx2_resize.long()].contiguous().unsqueeze(0)
                x2_crop_resize = F.interpolate(x2_crop, (x2_h, x2_w), mode='bilinear', align_corners=False)
                x2_ret.append(x2_crop_resize)

                crop_info = [xx1_resize, xx2_resize, yy1_resize, yy2_resize]
                crop_info_all.append(crop_info)
        return torch.cat(x2_ret, 0), crop_info_all

    def get_att_roi(self, att_mask, feature_stride, anchor_size, img_h, img_w, iou_thred=0.2, topk=1):
        """generation multi-leve ROIs upon spatial attention masks with NMS method"""
        device = att_mask.device
        with torch.no_grad():
            roi_ret_nms = []
            n, c, h, w = att_mask.size()
            att_corner_unmask = torch.zeros_like(att_mask).to(device)
            if self.num_classes == 200:
                att_corner_unmask[:, :, int(0.2 * h):int(0.8 * h), int(0.2 * w):int(0.8 * w)] = 1
            else:
                att_corner_unmask[:, :, int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)] = 1
            att_mask = att_mask * att_corner_unmask
            feat_anchor = generate_anchors_single_pyramid([anchor_size], [1], [h, w], feature_stride, 1).to(device)
            feat_new_cls = att_mask.clone()
            for i in range(n):
                boxes = feat_anchor.clone().float()
                scores = feat_new_cls[i].view(-1)
                score_thred_index = scores > scores.mean()
                boxes = boxes[score_thred_index, :]
                scores = scores[score_thred_index]
                # nms
                boxes_nms = pth_nms(torch.cat([boxes, scores.unsqueeze(1)], dim=1), iou_thred)[:topk]
                if len(boxes_nms.size()) == 1:
                    boxes_nms = boxes_nms.unsqueeze(0)
                # boxes_nms = pth_nms_merge(torch.cat([boxes, scores.unsqueeze(1)], dim=1), iou_thred, topk).to(device)
                boxes_nms[:, 0] = torch.clamp(boxes_nms[:, 0], min=0)
                boxes_nms[:, 1] = torch.clamp(boxes_nms[:, 1], min=0)
                boxes_nms[:, 2] = torch.clamp(boxes_nms[:, 2], max=img_w - 1)
                boxes_nms[:, 3] = torch.clamp(boxes_nms[:, 3], max=img_h - 1)
                roi_ret_nms.append(
                    torch.cat([torch.FloatTensor([i] * boxes_nms.size(0)).unsqueeze(1).to(device), boxes_nms], 1))

            return torch.cat(roi_ret_nms, 0)
    def forward(self, feats: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """
        这是TOODHead的前向传播函数，其主要作用是将来自上游网络的特征进行处理并生成目标检测的分类和边界框预测结果。
        函数中实现了任务对齐机制，该机制通过将特征映射到任务特定的空间，使得来自不同任务的特征能够更好地交互和共享信息。
        具体而言，1）函数接收来自上游网络的特征，通过循环遍历每个尺度下的特征，将每个特征图分别通过任务交互模块和任务分解模块进行处理，
        self.prior_generator.strides [(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)] 5个步长，尺度构建不同大小的框
        feats[0].shape
        Out[6]: torch.Size([2, 256, 96, 96])
        feats[1].shape
        Out[7]: torch.Size([2, 256, 48, 48])
        feats[2].shape
        Out[8]: torch.Size([2, 256, 24, 24])
        feats[3].shape
        Out[9]: torch.Size([2, 256, 12, 12])
        feats[4].shape
        Out[10]: torch.Size([2, 256, 6, 6])
        得到任务特定的特征表示。然后，分别通过分类和回归分支生成目标检测的分类和边界框预测结果。
        其中，分类分支的输出经过sigmoid几何平均函数，实现了分类结果的对齐；边界框分支的输出经过偏移量模块和变形采样模块，实现了边界框预测的对齐。
        最终，函数返回目标检测的分类和边界框预测结果。
        """
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Decoded box for all scale levels,
                    each is a 4D-tensor, the channels number is
                    num_anchors * 4. In [tl_x, tl_y, br_x, br_y] format.
        """
        cls_scores = []
        bbox_preds = []

        # stage I
        combined_feats = feats[0]
        # [f1_att, f2_att, f3_att, f4_att, f5_att, a1, a2, a3, a4, a5] = feats[1]
        # backbone_feats = feats[2]
        # apn_out = [f1_att, f2_att, f3_att, f4_att, f5_att, a1, a2, a3, a4, a5]
        # num_inputs = len(apn_out)

        # # Get the input image height and width for each feature map #
        # img_hs, img_ws = zip(*[(feat.size(2), feat.size(3)) for feat in combined_feats])#img_hs (84, 42, 21, 11，6) 最低层的84不要
        # img_h = 736
        # img_w = 736
        # # ROI operations
        # # roi pyramid # 这里对X2的后一层做ROI特征金字塔
        # roi_1 = self.get_att_roi(a1, 2 ** 1, 16, img_h, img_w, iou_thred=0.05, topk=9)#torch.Size([18, 6])
        # roi_2 = self.get_att_roi(a2, 2 ** 2, 32, img_h, img_w, iou_thred=0.05, topk=7)#torch.Size([14, 6])
        # roi_3 = self.get_att_roi(a3, 2 ** 3, 64, img_h, img_w, iou_thred=0.05, topk=5)#torch.Size([10, 6])
        # roi_4 = self.get_att_roi(a4, 2 ** 4, 128, img_h, img_w, iou_thred=0.05, topk=3)#torch.Size([4, 6])
        # roi_5 = self.get_att_roi(a5, 2 ** 5, 256, img_h, img_w, iou_thred=0.05, topk=1)#torch.Size([2, 6])
        # roi_list = [roi_1, roi_2, roi_3, roi_4, roi_5] # roi_list包含了不同尺度的ROI信息

        # roi_list = []
        # for i in range(num_inputs):
        #     scale = 2 ** (i + 3)
        #     feat_size = 64 * scale
        #     img_h, img_w = img_hs[i], img_ws[i]
        #     roi = self.get_att_roi(apn_out[i + 3], scale, feat_size, img_h, img_w, iou_thred=0.05,
        #                            topk=min(5, num_inputs - i))
        #     roi_list.append(roi)

        # # stage II
#        x2_crop_resize, _ = self.get_roi_crop_feat(backbone_feats[0], roi_list, 2 ** 1)
#        x3_crop_resize = self.get_roi_crop_feat(x2_crop_resize)
        # x4_crop_resize = self.get_roi_crop_feat(x3_crop_resize)
        # x5_crop_resize = self.get_roi_crop_feat(x4_crop_resize)
        # x3_crop_resize = self.layer3(x2_crop_resize)
        # x4_crop_resize = self.layer4(x3_crop_resize)

        # f3_crop_resize, f4_crop_resize, f5_crop_resize = fpn([x2_crop_resize, x3_crop_resize, x4_crop_resize])
        # f3_att_crop_resize, f4_att_crop_resize, f5_att_crop_resize, a3_crop_resize, a4_crop_resize, a5_crop_resize = self.apn(
        #     [f3_crop_resize, f4_crop_resize, f5_crop_resize])

        # Combine the outputs with original FPN outputs
        # f3_combined_crop_resize = f3_crop_resize + f3_att_crop_resize
        # f4_combined_crop_resize = f4_crop_resize + f4_att_crop_resize
        # f5_combined_crop_resize = f5_crop_resize + f5_att_crop_resize
        #
        # out_feats = tuple(f3_combined_crop_resize,f4_combined_crop_resize,f5_combined_crop_resize)
        # feats = out_feats

        # Combine the outputs with original FPN outputs 这里拿到了每层的注意力输出了其实 这个可以替换LayerAttention
        # combined_feats = [feat + apn_feat for feat, apn_feat in zip(feats, apn_out[:3])]
        # feats = combined_feats
        ##################

        for idx, (x, scale, stride) in enumerate(
                zip(combined_feats, self.scales, self.prior_generator.strides)):#feats->combined_feats
            b, c, h, w = x.shape
            anchor = self.prior_generator.single_level_grid_priors(
                (h, w), idx, device=x.device)
            anchor = torch.cat([anchor for _ in range(b)])
            # extract task interactive features
            inter_feats = []
            # 任务交互模块 inter_convs，也就是任务特征交互堆栈
            for inter_conv in self.inter_convs:
                x = inter_conv(x)
                inter_feats.append(x)
            feat = torch.cat(inter_feats, 1)
            # 任务分解模块
            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)
            reg_feat = self.reg_decomp(feat, avg_feat)
            # 分类对齐
            # cls prediction and alignment
            cls_logits = self.tood_cls(cls_feat)
            cls_prob = self.cls_prob_module(feat)
            cls_score = sigmoid_geometric_mean(cls_logits, cls_prob)#分类对齐使用了sigmoid_geometric_mean损失函数
            # 回归对齐
            # reg prediction and alignment
            if self.anchor_type == 'anchor_free':
                reg_dist = scale(self.tood_reg(reg_feat).exp()).float()
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
                reg_bbox = distance2bbox(#距离解码将距离(dist)解码为边界框预测(reg_bbox)
                    self.anchor_center(anchor) / stride[0],
                    reg_dist).reshape(b, h, w, 4).permute(0, 3, 1,
                                                          2)  # (b, c, h, w)
            elif self.anchor_type == 'anchor_based':
                reg_dist = scale(self.tood_reg(reg_feat)).float()
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
                reg_bbox = self.bbox_coder.decode(anchor, reg_dist).reshape(
                    b, h, w, 4).permute(0, 3, 1, 2) / stride[0]
            else:
                raise NotImplementedError(
                    f'Unknown anchor type: {self.anchor_type}.'
                    f'Please use `anchor_free` or `anchor_based`.')
            reg_offset = self.reg_offset_module(feat)
            bbox_pred = self.deform_sampling(reg_bbox.contiguous(),#bbox_pred是通过对reg_bbox采样得到的
                                             reg_offset.contiguous())
            # 通过deform_sampling函数进行了偏移量修正，修正偏移量使用了特征图上每个位置的偏移量，得到最终的预测边界框bbox_pred
            # After deform_sampling, some boxes will become invalid (The
            # left-top point is at the right or bottom of the right-bottom
            # point), which will make the GIoULoss negative.
            invalid_bbox_idx = (bbox_pred[:, [0]] > bbox_pred[:, [2]]) | \
                               (bbox_pred[:, [1]] > bbox_pred[:, [3]])
            invalid_bbox_idx = invalid_bbox_idx.expand_as(bbox_pred)
            bbox_pred = torch.where(invalid_bbox_idx, reg_bbox, bbox_pred)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return tuple(cls_scores), tuple(bbox_preds)

    def deform_sampling(self, feat: Tensor, offset: Tensor) -> Tensor:
        """Sampling the feature x according to offset.

        Args:
            feat (Tensor): Feature
            offset (Tensor): Spatial offset for feature sampling
        """
        # it is an equivalent implementation of bilinear interpolation
        b, c, h, w = feat.shape
        weight = feat.new_ones(c, 1, 1, 1)
        y = deform_conv2d(feat, offset, weight, 1, 0, 1, c, c)
        return y

    def anchor_center(self, anchors: Tensor) -> Tensor:
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            alignment_metrics: Tensor,
                            stride: Tuple[int, int]) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            alignment_metrics (Tensor): Alignment metrics with shape
                (N, num_total_anchors).
            stride (Tuple[int, int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = labels if self.epoch < self.initial_epoch else (
            labels, alignment_metrics)
        cls_loss_func = self.initial_loss_cls \
            if self.epoch < self.initial_epoch else self.loss_cls

        loss_cls = cls_loss_func(
            cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]

            # regression loss
            pos_bbox_weight = self.centerness_target(
                pos_anchors, pos_bbox_targets
            ) if self.epoch < self.initial_epoch else alignment_metrics[
                pos_inds]

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, alignment_metrics.sum(
        ), pos_bbox_weight.sum()

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """
        这段代码实现了检测模型中的损失函数计算。输入包括预测的类别得分和边界框坐标，以及真实的标注框和图像元信息等。
        首先，根据不同尺度的特征图大小和图像元信息计算得到每个尺度上的先验框，并根据先验框和真实标注框计算分类和回归目标。
        然后，对于每个尺度上的预测结果和目标计算对应的分类和回归损失，并返回总损失。
        最后，对于每个尺度上的损失值除以对应的平均因子以进行标准化。
        """
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)
        flatten_bbox_preds = torch.cat([
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) * stride[0]
            for bbox_pred, stride in zip(bbox_preds,
                                         self.prior_generator.strides)
        ], 1)

        cls_reg_targets = self.get_targets(
            flatten_cls_scores,
            flatten_bbox_preds,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         alignment_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox, \
            cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                alignment_metrics_list,
                self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: Optional[ConfigDict] = None,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """
        这段代码是目标检测模型中的一部分，用于将从网络头提取的特征转换为检测结果。这个函数接受多个输入参数，
        包括目标检测模型在不同尺度下得到的类别得分、边界框预测和先验框信息，以及一些配置和图像元信息。在函数内部，它遍历不同的尺度，
        并执行过滤、置信度阈值处理、NMS等操作，最终输出检测结果。该函数返回的结果包括检测到的边界框、对应的类别标签以及它们的得分。
        如果需要，还可以通过参数控制是否在原始图像上进行缩放。
        """
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (:obj:`ConfigDict`, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """

        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for cls_score, bbox_pred, priors, stride in zip(
                cls_score_list, bbox_pred_list, mlvl_priors,
                self.prior_generator.strides):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4) * stride[0]
            scores = cls_score.permute(1, 2,
                                       0).reshape(-1, self.cls_out_channels)

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bboxes = filtered_results['bbox_pred']

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = torch.cat(mlvl_bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def get_targets(self,
                    cls_scores: List[List[Tensor]],
                    bbox_preds: List[List[Tensor]],
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True) -> tuple:
        """
        这段代码实现了基于Faster R-CNN的目标检测中的训练目标生成过程。具体来说，该函数计算了用于训练模型的回归和分类目标，
        以用于指导模型在训练过程中学习正确的检测结果。该函数接受多个输入参数，包括分类预测分数，边界框预测结果，锚点框，
        有效标志等信息，以及图像元数据和实例数据等信息。然后它通过调用内部函数来计算每个图像的目标，得到每个级别的锚点、标签、标签权重、边界框目标和标准化对齐度量，并返回这些值的列表。
        总的来说，这段代码实现了目标检测中一个重要的步骤，即计算训练目标，使模型能够在训练过程中不断优化以更好地检测出目标。
        """
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores (list[list[Tensor]]): Classification predictions of
                images, a 3D-Tensor with shape [num_imgs, num_priors,
                num_classes].
            bbox_preds (list[list[Tensor]]): Decoded bboxes predictions of one
                image, a 3D-Tensor with shape [num_imgs, num_priors, 4] in
                [tl_x, tl_y, br_x, br_y] format.
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: a tuple containing learning targets.

                - anchors_list (list[list[Tensor]]): Anchors of each level.
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - norm_alignment_metrics_list (list[Tensor]): Normalized
                  alignment metrics of each level.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        # anchor_list: list(b * [-1, 4])

        # get epoch information from message hub
        message_hub = MessageHub.get_current_instance()
        self.epoch = message_hub.get_info('epoch')

        if self.epoch < self.initial_epoch:
            (all_anchors, all_labels, all_label_weights, all_bbox_targets,
             all_bbox_weights, pos_inds_list, neg_inds_list,
             sampling_result) = multi_apply(
                 super()._get_targets_single,
                 anchor_list,
                 valid_flag_list,
                 num_level_anchors_list,
                 batch_gt_instances,
                 batch_img_metas,
                 batch_gt_instances_ignore,
                 unmap_outputs=unmap_outputs)
            all_assign_metrics = [
                weight[..., 0] for weight in all_bbox_weights
            ]
        else:
            (all_anchors, all_labels, all_label_weights, all_bbox_targets,
             all_assign_metrics) = multi_apply(
                 self._get_targets_single,
                 cls_scores,
                 bbox_preds,
                 anchor_list,
                 valid_flag_list,
                 batch_gt_instances,
                 batch_img_metas,
                 batch_gt_instances_ignore,
                 unmap_outputs=unmap_outputs)

        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        norm_alignment_metrics_list = images_to_levels(all_assign_metrics,
                                                       num_level_anchors)

        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, norm_alignment_metrics_list)

    """
      这是一个用于目标检测模型训练的函数。函数的输入包括cls_scores（检测框的分数），
      bbox_preds（检测框的位置预测），flat_anchors（不同尺度的锚框），valid_flags（有效的锚框标志），
      gt_instances（真实目标实例数据），img_meta（当前图像的元信息），gt_instances_ignore（不参与训练的实例数据），
      以及unmap_outputs（是否需要将输出映射回原始锚点集）。函数的输出包括anchors（图像中的所有锚点），labels（所有锚点的标签），
      label_weights（所有锚点的标签权重），bbox_targets（所有锚点的位置目标），norm_alignment_metrics（
      所有锚点的规范化对齐度量）。该函数通过对锚点进行正负样本的分配，并计算位置目标和对齐度量，
      从而生成用于训练目标检测模型的数据。

      """
    def _get_targets_single(self,
                            cls_scores: Tensor,
                            bbox_preds: Tensor,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression, classification targets for anchors in a single
        image.
        
        Args:
            cls_scores (Tensor): Box scores for each image.
            bbox_preds (Tensor): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                norm_alignment_metrics (Tensor): Normalized alignment metrics
                    of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors #正负样本的分配
        anchors = flat_anchors[inside_flags, :]
        pred_instances = InstanceData(
            priors=anchors,
            scores=cls_scores[inside_flags, :],
            bboxes=bbox_preds[inside_flags, :])
        # ATSS机制实现的部分在alignment_assigner.assign()函数中，其中调用了ATSSLoss的assign()方法，
        # 根据锚框和GT实例之间的IoU和位置信息计算得分，使用ATSS方法分配正负样本，并计算对齐度量。TaskAlignedAssigner
        assign_result = self.alignment_assigner.assign(pred_instances,
                                                       gt_instances,
                                                       gt_instances_ignore,
                                                       self.alpha, self.beta)
        # 后面需要根据分配的iou结果来加上对比损失计算
        assign_ious = assign_result.max_overlaps#计算对齐结果
        assign_metrics = assign_result.assign_metrics
        # 在这里使用sampler对分配结果进行采样，得到正样本和负样本的索引。根据正负样本的索引，为每个样本计算分类目标、回归目标和对齐度量。
        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        norm_alignment_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            pos_alignment_metrics = assign_metrics[gt_class_inds]#位置对其
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            norm_alignment_metrics = unmap(norm_alignment_metrics,
                                           num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets,
                norm_alignment_metrics)
