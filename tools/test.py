# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.registry import RUNNERS
from mmdet.utils import add_dump_metric, register_all_modules
"""
/hy-tmp/mmdetection/configs/yolox/yolox_s_8xb8-300e_coco-solar-cell.py
/hy-tmp/mmdetection/tools/work_dirs/yolox_s_8xb8-300e_coco_solar_cell/epoch_300.pth
--out /hy-tmp/mmdetection/tools/work_dirs/yolox_s_8xb8-300e_coco_solar_cell/test.pkl
--show-dir /hy-tmp/mmdetection/tools/work_dirs/yolox_s_8xb8-300e_coco_solar_cell/test_results

# 输出test.pkl
/hy-tmp/mmdetection/configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco_solar_cell.py
/hy-tmp/mmdetection/tools/work_dirs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco_cbam/epoch_120.pth
--out /hy-tmp/mmdetection/tools/work_dirs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco_cbam/test.pkl
--show-dir /hy-tmp/mmdetection/tools/work_dirs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco_cbam/test_results

/hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-solar-cell.py
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell_new_data/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell_new_data_and_label_epoch_21.pth
--out /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell_new_data/test.pkl
--show-dir /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell_new_data/test_results4

/hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-solar-cell.py
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_new_solar_data/epoch_24.pth
--out /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_new_solar_data/test.pkl
--show-dir /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_new_solar_data/test_results
 Epoch(test) [100/139]    eta: 0:00:12  time: 0.3173  data_time: 0.2492  memory: 454 
 
/hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-solar-cell.py
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone/epoch_23.pth
--out /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone/test.pkl
--show-dir /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone/test_results

/root/autodl-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-solar-cell-EL-audodl.py
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_EL-image-ALBU-RandomChoice/epoch_21.pth
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_EL-image-ALBU-RandomChoice/test.pkl
--show-dir /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_EL-image-ALBU-RandomChoice/test_results


/hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-hengyuanyun.py
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin/epoch_21.pth
--out /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin/test.pkl
--show-dir /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin/test_results

/root/autodl-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-autodl.py
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-paapn-data3-process-orin-bs4/epoch_21.pth
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-paapn-data3-process-orin-bs4/test.pkl
--show-dir /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-paapn-data3-process-orin-bs4/test_results


/root/autodl-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-hengyuanyun-orin-bs4.py



/root/autodl-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin.py
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin/epoch_21.pth
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin/test.pkl
--show-dir /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin/test_results
python tools/test.py /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-acris-APFPN-CLAFE-HCBP.py


/root/autodl-tmp/mmdetection/configs/ours/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-acris-APFPN-CLAFE-HCBP.py
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/epoch_22.pth
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/test.pkl
--show-dir /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/test_results_large


/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin.py
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/epoch_24_orin_APN.pth
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/test.pkl
--show-dir /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/test_results

# 可视化热力图
/root/mmdetection-3.x/configs/tood/Focus.py
/root/autodl-tmp/mmdetection/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/epoch_22.pth
--out /root/autodl-tmp/mmdetection/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/test.pkl
--show-dir /root/autodl-tmp/mmdetection/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/test_results
python /root/mmdetection-3.x/tools/analysis_tools/get_flops.py /root/mmdetection-3.x/configs/ours/rtmdet/rtmdet_s_8xb32-300e_EL.py
python tools/test.py /root/mmdetection-3.x/configs/ours/rtmdet/rtmdet_s_8xb32-300e_EL.py /root/mmdetection-3.x/work_dirs/rtmdet_s_8xb32-300e_EL/epoch_300.pth --out /root/mmdetection-3.x/work_dirs/rtmdet_s_8xb32-300e_EL/test.pkl --show-dir /root/mmdetection-3.x/work_dirs/rtmdet_s_8xb32-300e_EL/test_results


# 重新做对比可视化图
/root/mmdetection-3.x/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin.py
/root/mmdetection-3.x/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/epoch_24.pth
--out /root/mmdetection-3.x/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/test.pkl
--show-dir /root/mmdetection-3.x/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/test_results

"""
### 生成results.bbox.json

# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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


def main():
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

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    # Dump predictions
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        add_dump_metric(args, cfg)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test()
"""
### linux 输出 test.pkl 文件 1)
/hy-tmp/mmdetection/configs/deformable_detr/linux-mm3-deformable-detr_r50_16xb2-50e_coco.py
/hy-tmp/mmdetection/tools/work_dirs/deformable-detr_r50_16xb2-50e_coco/epoch_270.pth
--out /hy-tmp/mmdetection/tools/work_dirs/deformable-detr_r50_16xb2-50e_coco/test_270.pkl
--show-dir /hy-tmp/mmdetection/tools/work_dirs/deformable-detr_r50_16xb2-50e_coco/test_result

### linux 输出 results.bbox.json 文件 3)
/hy-tmp/mmdetection/configs/deformable_detr/linux-mm3-deformable-detr_r50_16xb2-50e_coco.py
/hy-tmp/mmdetection/tools/work_dirs/deformable-detr_r50_16xb2-50e_coco/epoch_270.pth
--format-only
--options "jsonfile_prefix=/hy-tmp/mmdetection-master/tools/work_dirs/20221022_linux3090_deformable_detr_r50_16x2_50e_neu_det/results_bbox"

"""

if __name__ == '__main__':
    main()
