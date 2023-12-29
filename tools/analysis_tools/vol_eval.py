from argparse import ArgumentParser

import mmcv
import mmengine
from mmdet import datasets
from mmdet.evaluation.functional import eval_map #修改导入


def voc_eval(result_file, dataset, iou_thr=0.5, nproc=4):
    det_results = mmcv.load(result_file)
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    eval_map(
        det_results,
        annotations,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        logger='print',
        nproc=nproc)


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    parser.add_argument(
        '--nproc',
        type=int,
        default=4,
        help='Processes to be used for computing mAP')
    args = parser.parse_args()
    # cfg = mmcv.Config.fromfile(args.config)
    # test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    cfg = mmengine.Config.fromfile(args.config)
    test_dataset = mmengine.runner.obj_from_dict(cfg.data.test, datasets)
    voc_eval(args.result, test_dataset, args.iou_thr, args.nproc)

"""
ubuntu
../work_dirs/20221019_deformable_detr_r50_16x2_50e_neu_det/test.pkl
F:\\workspace\\pytorch\\mmdetection-master\\configs\\deformable_detr\\20220929_deformable_detr_r50_16x2_50e_coco.py 
linux
利用test.pkl  输出每类精度 2)
/hy-tmp/mmdetection/tools/work_dirs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco/test.pkl
/hy-tmp/mmdetection/configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco_solar_cell.py

/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/test.pkl
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-acris-APFPN-CLAFE-HCBP.py

"""
if __name__ == '__main__':
    main()