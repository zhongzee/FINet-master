# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import sys
curPath = os.path.abspath(os.path.dirname(__file__ ))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import register_all_modules
import torch
import numpy as np
import random
# /hy-tmp/mmdetection/configs/deformable_detr/20221028_mm3x_linux3090_deformable_detr_r50_16x2_50e_coco.py
# python tools/train.py hy-tmp/mmdetection/configs/deformable_detr/20221031-linux-mm3-deformable-detr_r50_16xb2-50e_coco.py --resume
# /hy-tmp/mmdetection/configs/deformable_detr/linux-mm3-deformable-detr_r50_16xb2-50e_coco.py --resume (这里 load_from=None)
# python tools/train.py /hy-tmp/mmdetection/configs/soft_teacher/soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.01-coco.py
# /hy-tmp/mmdetection/configs/soft_teacher/my-soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py
# /hy-tmp/mmdetection/configs/soft_teacher/nodule-soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py
# 最新
# /hy-tmp/mmdetection/configs/deformable_detr/20230227-linux-mm3-deformable-detr_r50_16xb2-50e_solar_cell_coco.py
# /hy-tmp/mmdetection/configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco_solar_cell.py
# /hy-tmp/mmdetection/configs/yolo/yolov3_d53_8xb8-320-273e_coco_solar_cell.py

# /hy-tmp/mmdetection/configs/yolox/yolox_s_8xb8-300e_coco.py
# /hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-solar-cell.py

# /hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-neu-det.py

# python tools/train.py /root/autodl-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-solar-cell-EL-audodl-orin.py --resume
# python tools/train.py /root/autodl-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-audodl.py --resume
# python tools/train.py /hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-hengyuanyun-orin-bs4.py --resume
# python tools/train.py /hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-hengyuanyun.py --resume
# python tools/train.py /hy-tmp/mmdetection/configs/tood/tood_r50_fpn_ms-2x_coco_multi_backbone-EL-image-apn-hengyuanyun.py
# python tools/train.py /hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-acmix-hengyuanyun.py
# python tools/train.py /hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-hyy-orin-bs4-aug.py
# python tools/train.py /hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-solar-cell-EL-audodl-orin-NASFCOS_FPN.py --resume
# python tools/train.py /hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-paapn-hengyuanyun.py
# python tools/train.py /hy-tmp/mmdetection/configs/tood/tood_swin-B-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-hengyuanyun.py --resume

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    set_seed(88)
    args = parse_args()

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()

# ssh -p 23916 root@i-2.gpushare.com
if __name__ == '__main__':
    main()
