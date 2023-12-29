# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptInstanceList
from ..losses import smooth_l1_loss
from ..task_modules.samplers import PseudoSampler
from ..utils import multi_apply
from .anchor_head import AnchorHead
import numpy as np
import torch
from torch import nn
from torch.nn import init

import torch
from torch import nn
import math
# RM降维度
# 生成正交矩阵
# def rand_orthonormal_matrix(n):
#     # 生成随机矩阵
#     h = torch.randn(n, n)
#     # 对随机矩阵进行QR分解，得到正交矩阵
#     q, r = torch.qr(h)
#     return q
#
#
# # 生成投影矩阵
# def rand_projection_matrix(n=16384, m=128):
#     # 生成正交矩阵
#     orth = rand_orthonormal_matrix(n)
#     # 取正交矩阵的前m列作为投影矩阵
#     proj = orth[:, :m]
#     return proj

# 降维
class RM(nn.Module):
    def __init__(self, target_dim=128):
        super().__init__()
        # self.input_size = 16384
        self.target_dim = target_dim

    def forward(self, x):
        batch_size, in_dim = x.shape
        # 随机生成目标维度的投影矩阵，每个元素取自 N(0,1) 分布
        rand_matrix = torch.randn((in_dim, self.target_dim), device=x.device)
        # 对投影矩阵进行QR分解，得到Q矩阵，Q^T*Q=I
        q_matrix, _ = torch.qr(rand_matrix)
        # 将输入矩阵投影到低维空间
        projected = x @ q_matrix # torch.Size([24, 128])
        z = torch.reshape(projected, (batch_size, self.target_dim, 1, 1))
        return z # torch.Size([24, 128, 1, 1])

# ssd_head_bcnn_cbam_senet.py
class BilinearPooling(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)
        # super().__init__()

    def forward(self, x, y):
        batch_size = x.size(0)
        channel_size = x.size(1)
        feature_size = x.size(2) * x.size(3)
        x = x.view(batch_size, channel_size, feature_size)
        y = y.view(batch_size, channel_size, feature_size)
        x = torch.bmm(x, torch.transpose(y, 1, 2)) / feature_size #其实就是将backbone提取出来的特征 自己与自己相乘,这里改成x和y做外积

        x = x.view(batch_size, -1)
        x = torch.sqrt(x + 1e-5)

        # x = x.view(batch_size, -1)
        # x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-10)

        x = torch.nn.functional.normalize(x) # 这里输出是torch.Size([24, 16384]) 需要降维度 或者改变维度
        return x

class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# if __name__ == '__main__':
#     input = torch.randn(50, 512, 7, 7)
#     se = SEAttention(channel=512, reduction=8)
#     output = se(input)
#     print(output.shape)


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
        result = torch.cat([max_result, avg_result], 1)#torch.Size([24, 1, 1, 1])
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
        b, c, _, _ = x.size()#看下输入 128，24，_,_
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


# if __name__ == '__main__':
#     input = torch.randn(50, 512, 7, 7)
#     kernel_size = input.shape[2]
#     cbam = CBAMBlock(channel=512, reduction=16, kernel_size=kernel_size)
#     output = cbam(input)
#     print(output.shape)


# TODO: add loss evaluator for SSD
@MODELS.register_module()
class SSDHead(AnchorHead):
    """Implementation of `SSD head <https://arxiv.org/abs/1512.02325>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Sequence[int]): Number of channels in the input feature
            map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Defaults to 0.
        feat_channels (int): Number of hidden channels when stacked_convs
            > 0. Defaults to 256.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, Optional): Dictionary to construct
            and config conv layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, Optional): Dictionary to construct
            and config norm layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, Optional): Dictionary to construct
            and config activation layer. Defaults to None.
        anchor_generator (:obj:`ConfigDict` or dict): Config dict for anchor
            generator.
        bbox_coder (:obj:`ConfigDict` or dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Defaults to False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        train_cfg (:obj:`ConfigDict` or dict, Optional): Training config of
            anchor head.
        test_cfg (:obj:`ConfigDict` or dict, Optional): Testing config of
            anchor head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], Optional): Initialization config dict.
    """  # noqa: W605

    def __init__(
        self,
        num_classes: int = 80,
        in_channels: Sequence[int] = (512, 1024, 512, 256, 256, 256),
        stacked_convs: int = 0,
        feat_channels: int = 256,
        use_depthwise: bool = False,
        conv_cfg: Optional[ConfigType] = None,
        norm_cfg: Optional[ConfigType] = None,
        act_cfg: Optional[ConfigType] = None,
        anchor_generator: ConfigType = dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=300,
            strides=[8, 16, 32, 64, 100, 300],
            ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
            basesize_ratio_range=(0.1, 0.9)),
        bbox_coder: ConfigType = dict(
            type='DeltaXYWHBBoxCoder',
            clip_border=True,
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        reg_decoded_bbox: bool = False,
        train_cfg: Optional[ConfigType] = None,
        test_cfg: Optional[ConfigType] = None,
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform', bias=0)
    ) -> None:
        super(AnchorHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.stacked_convs = stacked_convs
        self.feat_channels = feat_channels
        self.use_depthwise = use_depthwise
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.cls_out_channels = num_classes + 1  # add background class
        self.prior_generator = TASK_UTILS.build(anchor_generator)

        # Usually the numbers of anchors for each level are the same
        # except SSD detectors. So it is an int in the most dense
        # heads but a list of int in SSDHead
        self.num_base_priors = self.prior_generator.num_base_priors

        self._init_layers()

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'], default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)

        # 增加cbam注意力机制
        #self.cbam = CBAMBlock(channel=512, reduction=16, kernel_size=49) #参数暂用默认的
        self.cbam = CBAMBlock(channel=128, reduction=16, kernel_size=49)
        # 增加senet注意力机制
        self.senet = SEAttention(channel=128, reduction=16)
        # BilinearPooling
        self.bilinear_pooling = BilinearPooling()
        # RM降维
        self.rm = RM()
    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # TODO: Use registry to choose ConvModule type
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule

        for channel, num_base_priors in zip(self.in_channels,
                                            self.num_base_priors):
            cls_layers = []
            reg_layers = []
            in_channel = channel
            # build stacked conv tower, not used in default ssd
            for i in range(self.stacked_convs):
                cls_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    conv(
                        in_channel,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                in_channel = self.feat_channels
            # SSD-Lite head
            if self.use_depthwise:
                cls_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                reg_layers.append(
                    ConvModule(
                        in_channel,
                        in_channel,
                        3,
                        padding=1,
                        groups=in_channel,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            cls_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * self.cls_out_channels,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            reg_layers.append(
                nn.Conv2d(
                    in_channel,
                    num_base_priors * 4,
                    kernel_size=1 if self.use_depthwise else 3,
                    padding=0 if self.use_depthwise else 1))
            self.cls_convs.append(nn.Sequential(*cls_layers))
            self.reg_convs.append(nn.Sequential(*reg_layers))

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[list[Tensor], list[Tensor]]: A tuple of cls_scores list and
            bbox_preds list.

            - cls_scores (list[Tensor]): Classification scores for all scale \
            levels, each is a 4D-tensor, the channels number is \
            num_anchors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale \
            levels, each is a 4D-tensor, the channels number is \
            num_anchors * 4.
        """
        cls_scores = []
        bbox_preds = []
        # 对最后一层添加cbam注意力机制
        pred_outs = x[:-1]
        cbam_outs = self.cbam(x[len(x)-1])
        # 也对倒数第1层添加senet注意力机制
        senet_outs = self.senet(x[len(x)-1])#torch.Size([24, 128, 1, 1])
        # 然后对两层输出进行BilinearPooling
        last_outs = self.bilinear_pooling(cbam_outs,senet_outs)#torch.Size([24, 16384])
        # 紧凑的双线性模型
        # 降维 (这里是对第6层分别使用不同的注意机制，然后进行外积再降维，构成最后一层输出)
        rm_outs = self.rm(last_outs)
        # 返回tuple类型
        x = tuple(list(pred_outs)+[rm_outs])
        # x = [x for x in self.cbam(x)]
        for feat, reg_conv, cls_conv in zip(x, self.reg_convs, self.cls_convs):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))
        return cls_scores, bbox_preds

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            anchor: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor,
                            avg_factor: int) -> Tuple[Tensor, Tensor]:
        """Compute loss of a single image.

        Args:
            cls_score (Tensor): Box scores for eachimage
                Has shape (num_total_anchors, num_classes).
            bbox_pred (Tensor): Box energies / deltas for each image
                level with shape (num_total_anchors, 4).
            anchors (Tensor): Box reference for each scale level with shape
                (num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (num_total_anchors,).
            label_weights (Tensor): Label weights of each anchor with shape
                (num_total_anchors,)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            Tuple[Tensor, Tensor]: A tuple of cls loss and bbox loss of one
            feature map.
        """

        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(
            as_tuple=False).reshape(-1)
        neg_inds = (labels == self.num_classes).nonzero(
            as_tuple=False).view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.train_cfg['neg_pos_ratio'] * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / avg_factor

        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)

        loss_bbox = smooth_l1_loss(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=self.train_cfg['smoothl1_beta'],
            avg_factor=avg_factor)
        return loss_cls[None], loss_bbox

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, List[Tensor]]:
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
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
            dict[str, list[Tensor]]: A dictionary of loss components. the dict
            has components below:

            - loss_cls (list[Tensor]): A list containing each feature map \
            classification loss.
            - loss_bbox (list[Tensor]): A list containing each feature map \
            regression loss.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            unmap_outputs=True)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor) = cls_reg_targets

        num_images = len(batch_img_metas)
        all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels) for s in cls_scores
        ], 1)
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        all_label_weights = torch.cat(label_weights_list,
                                      -1).view(num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        # concat all level anchors to a single tensor
        all_anchors = []
        for i in range(num_images):
            all_anchors.append(torch.cat(anchor_list[i]))

        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            all_cls_scores,
            all_bbox_preds,
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            avg_factor=avg_factor)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
