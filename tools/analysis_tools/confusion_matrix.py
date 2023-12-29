import argparse
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv.ops import nms
from mmengine import Config, DictAction
from mmengine.fileio import load
from mmengine.utils import ProgressBar

from mmdet.evaluation import bbox_overlaps
from mmdet.registry import DATASETS
from mmdet.utils import (register_all_modules, replace_cfg_vals,
                         update_data_root)

"""
/hy-tmp/mmdetection/configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco_solar_cell.py
/hy-tmp/mmdetection/tools/work_dirs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco_cbam/test.pkl
/hy-tmp/mmdetection/tools/work_dirs/ssdlite_mobilenetv2-scratch_8xb24-600e_coco_cbam/
--show

/hy-tmp/mmdetection/configs/yolox/yolox_s_8xb8-300e_coco-solar-cell.py
/hy-tmp/mmdetection/tools/work_dirs/yolox_s_8xb8-300e_coco_solar_cell/test.pkl
/hy-tmp/mmdetection/tools/work_dirs/yolox_s_8xb8-300e_coco_solar_cell/
--show

/hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-solar-cell.py
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell/test.pkl
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell/
--show

/hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-solar-cell.py
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell_new_data/test.pkl
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell_new_data/
--show

/hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-solar-cell.py
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_new_solar_data/test.pkl
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_new_solar_data/

/hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-solar-cell.py
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone/test.pkl
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone/

/root/autodl-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco-solar-cell-EL-audodl.py
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_EL-image-ALBU-RandomChoice/test.pkl
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_EL-image-ALBU-RandomChoice/

/hy-tmp/mmdetection/configs/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-hengyuanyun.py
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin/test.pkl
/hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin/

/root/autodl-tmp/mmdetection/configs/ours/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-acris-APFPN-CLAFE-HCBP.py
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/test.pkl
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/

/root/autodl-tmp/mmdetection/configs/ours/tood/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-acris-APFPN-CLAFE-HCBP.py
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/test-0.5.pkl
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/

/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin.py
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/epoch_24_orin_APN.pth
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/
"""
def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        'save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='plasma',
        help='theme of the matrix color map')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold to filter detection bboxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to be considered as matched')
    parser.add_argument(
        '--nms-iou-thr',
        type=float,
        default=None,
        help='nms IoU threshold, only applied when users want to change the'
        'nms IoU threshold.')
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
    args = parser.parse_args()
    return args


def calculate_confusion_matrix(dataset,
                               results,
                               score_thr=0,#默认0
                               nms_iou_thr=None,
                               tp_iou_thr=0.5):#默认0.5
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[ndarray]): A list of detection results in each image.
        score_thr (float|optional): Score threshold to filter bboxes.
            Default: 0.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
        tp_iou_thr (float|optional): IoU threshold to be considered as matched.
            Default: 0.5.
    """
    num_classes = len(dataset.metainfo['CLASSES'])
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)
    prog_bar = ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res['pred_instances'] #需要知道这个
        gts = dataset.get_data_info(idx)['instances']
        analyze_per_img_dets(confusion_matrix, gts, res_bboxes, score_thr,
                             tp_iou_thr, nms_iou_thr)
        prog_bar.update()
    return confusion_matrix


def analyze_per_img_dets(confusion_matrix,
                         gts,
                         result,
                         score_thr=0,#默认0
                         tp_iou_thr=0.5, #默认是0.5
                         nms_iou_thr=None):
    """Analyze detection results on each image.

    Args:
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
        gt_bboxes (ndarray): Ground truth bboxes, has shape (num_gt, 4).
        gt_labels (ndarray): Ground truth labels, has shape (num_gt).
        result (ndarray): Detection results, has shape
            (num_classes, num_bboxes, 5).
        score_thr (float): Score threshold to filter bboxes.
            Default: 0.
        tp_iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
    """
    true_positives = np.zeros(len(gts))
    gt_bboxes = []
    gt_labels = []
    for gt in gts:
        gt_bboxes.append(gt['bbox'])
        gt_labels.append(gt['bbox_label'])

    gt_bboxes = np.array(gt_bboxes)
    gt_labels = np.array(gt_labels)

    unique_label = np.unique(result['labels'].numpy())

    for det_label in unique_label:#需要知道这个
        mask = (result['labels'] == det_label)
        det_bboxes = result['bboxes'][mask].numpy()
        det_scores = result['scores'][mask].numpy()

        if nms_iou_thr:
            det_bboxes, _ = nms(
                det_bboxes, det_scores, nms_iou_thr, score_threshold=score_thr)
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        for i, score in enumerate(det_scores):
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # TP
                        confusion_matrix[gt_label, det_label] += 1
                if det_match == 0:  # BG FP
                    confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:  # FN
            confusion_matrix[gt_label, -1] += 1


# def plot_confusion_matrix(confusion_matrix,
#                           labels,
#                           save_dir=None,
#                           show=True,
#                           title='Confusion Matrix',
#                           color_theme='plasma'):
#     """Draw confusion matrix with matplotlib.
#
#     Args:
#         confusion_matrix (ndarray): The confusion matrix.
#         labels (list[str]): List of class names.
#         save_dir (str|optional): If set, save the confusion matrix plot to the
#             given path. Default: None.
#         show (bool): Whether to show the plot. Default: True.
#         title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
#         color_theme (str): Theme of the matrix color map. Default: `plasma`.
#     """
#     # normalize the confusion matrix
#     per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
#     confusion_matrix = \
#         confusion_matrix.astype(np.float32) / per_label_sums * 100
#
#     num_classes = len(labels)
#     fig, ax = plt.subplots(
#         figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=180)
#     cmap = plt.get_cmap(color_theme)
#     im = ax.imshow(confusion_matrix, cmap=cmap)
#     plt.colorbar(mappable=im, ax=ax)
#
#     title_font = {'weight': 'bold', 'size': 12}
#     ax.set_title(title, fontdict=title_font)
#     label_font = {'size': 10}
#     plt.ylabel('Ground Truth Label', fontdict=label_font)
#     plt.xlabel('Prediction Label', fontdict=label_font)
#
#     # draw locator
#     xmajor_locator = MultipleLocator(1)
#     xminor_locator = MultipleLocator(0.5)
#     ax.xaxis.set_major_locator(xmajor_locator)
#     ax.xaxis.set_minor_locator(xminor_locator)
#     ymajor_locator = MultipleLocator(1)
#     yminor_locator = MultipleLocator(0.5)
#     ax.yaxis.set_major_locator(ymajor_locator)
#     ax.yaxis.set_minor_locator(yminor_locator)
#
#     # draw grid
#     ax.grid(True, which='minor', linestyle='-')
#
#     # draw label
#     ax.set_xticks(np.arange(num_classes))
#     ax.set_yticks(np.arange(num_classes))
#     ax.set_xticklabels(labels)
#     ax.set_yticklabels(labels)
#
#     ax.tick_params(
#         axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
#     plt.setp(
#         ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
#
#     # draw confution matrix value
#     for i in range(num_classes):
#         for j in range(num_classes):
#             ax.text(
#                 j,
#                 i,
#                 '{}%'.format(
#                     int(confusion_matrix[
#                         i,
#                         j]) if not np.isnan(confusion_matrix[i, j]) else -1),
#                 ha='center',
#                 va='center',
#                 color='w',
#                 size=7)
#
#     ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1
#
#     fig.tight_layout()
#     if save_dir is not None:
#         plt.savefig(
#             os.path.join(save_dir, 'confusion_matrix.png'), format='png')
#     if show:
#         plt.show()

# def plot_confusion_matrix(confusion_matrix,
#                           labels,
#                           save_dir=None,
#                           show=True,
#                           normalize=True,
#                           title='Confusion Matrix',
#                           cmap='Blues'):
#     """Draw confusion matrix with seaborn.
#
#     Args:
#         confusion_matrix (ndarray): The confusion matrix.
#         labels (list[str]): List of class names.
#         save_dir (str|optional): If set, save the confusion matrix plot to the
#             given path. Default: None.
#         show (bool): Whether to show the plot. Default: True.
#         normalize (bool): Whether to normalize the confusion matrix. Default: True.
#         title (str): Title of the plot. Default: 'Confusion Matrix'.
#         cmap (str): The mapping from data values to color space. Default: 'Blues'.
#     """
#     import seaborn as sn
#
#     # normalize the confusion matrix
#     if normalize:
#         per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
#         confusion_matrix = \
#             confusion_matrix.astype(np.float32) / per_label_sums  # normalize columns
#     confusion_matrix[confusion_matrix < 0.005] = np.nan  # don't annotate (would appear as 0.00)
#
#     fig, ax = plt.subplots(figsize=(12, 9), tight_layout=True)
#     sn.set(font_scale=1.2)  # for label size
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
#         sn.heatmap(confusion_matrix,
#                    ax=ax,
#                    annot=True,
#                    annot_kws={'size': 8},
#                    cmap=cmap,
#                    fmt='.2f',
#                    square=True,
#                    xticklabels=labels,
#                    yticklabels=labels).set_facecolor((1, 1, 1))
#     ax.set_xlabel('True')
#     ax.set_ylabel('Predicted')
#     ax.set_title(title)
#     if save_dir is not None:
#         fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=600)
#     if show:
#         plt.show()
#     else:
#         plt.close(fig)

def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir=None,
                          show=True,
                          normalize=True,
                          title='Confusion Matrix',
                          cmap='Blues'):
    """Draw confusion matrix with seaborn.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        normalize (bool): Whether to normalize the confusion matrix. Default: True.
        title (str): Title of the plot. Default: 'Confusion Matrix'.
        cmap (str): The mapping from data values to color space. Default: 'Blues'.
    """
    import seaborn as sn
    from matplotlib.font_manager import FontProperties

    # Define font properties
    title_font = FontProperties(family='Times New Roman', style='normal', size=20, weight='bold', stretch='normal')
    label_font = FontProperties(family='Times New Roman', style='normal', size=14, weight='bold', stretch='normal')
    ticks_font = FontProperties(family='Times New Roman', style='normal', size=10, weight='bold', stretch='normal')

    # normalize the confusion matrix
    if normalize:
        per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
        confusion_matrix = \
            confusion_matrix.astype(np.float32) / per_label_sums  # normalize columns
    confusion_matrix[confusion_matrix < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    fig, ax = plt.subplots(figsize=(12, 9), dpi=300, tight_layout=True)
    sn.set(font_scale=1.2)  # for label size
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
        sn.heatmap(confusion_matrix,
                   ax=ax,
                   annot=True,
                   annot_kws={'size': 8, 'fontproperties': ticks_font},
                   cmap=cmap,
                   fmt='.2f',
                   square=True,
                   xticklabels=labels,
                   yticklabels=labels).set_facecolor((1, 1, 1))
    ax.set_xlabel('True', fontproperties=label_font)
    ax.set_ylabel('Predicted', fontproperties=label_font)
    ax.set_title(title, fontproperties=title_font)
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=ticks_font, rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=ticks_font)
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'confusion_matrix_0.45.png'), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    register_all_modules()
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    results = load(args.prediction_path)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    confusion_matrix = calculate_confusion_matrix(dataset, results,
                                                  args.score_thr,
                                                  args.nms_iou_thr,
                                                  args.tp_iou_thr)
    # plot_confusion_matrix(
    #     confusion_matrix,
    #     dataset.metainfo['CLASSES'] + ('background', ),
    #     save_dir=args.save_dir,
    #     show=args.show,
    #     color_theme=args.color_theme)
    plot_confusion_matrix(
        confusion_matrix,
        dataset.metainfo['CLASSES'] + ('background',),
        save_dir=args.save_dir,
        show=args.show)


if __name__ == '__main__':
    main()
