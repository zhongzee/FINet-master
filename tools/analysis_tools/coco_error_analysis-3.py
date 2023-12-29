# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from argparse import ArgumentParser
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.font_manager
# matplotlib.font_manager._rebuild()
# plt.rcParams['font.family'] = 'Times New Roman'



def hex_to_rgb(hex_color):
    # remove '#' if it is present
    if hex_color[0] == '#':
        hex_color = hex_color[1:]
    # convert hex to rgb
    return [int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4)]

# Here is the modified `makeplot` function to plot PR curves for all classes in one figure:

# def makeplot(rs, ps, outDir, class_name, iou_type, ax):
#     cs = np.vstack([
#         np.array(hex_to_rgb('FFC9C9')),  # light red for C75
#         np.array(hex_to_rgb('FF914D')),  # light blue for C50
#         np.array([0.31, 0.51, 0.74]),
#         np.array([0.75, 0.31, 0.30]),
#         np.array([0.36, 0.90, 0.38]),
#         np.array([0.50, 0.39, 0.64]),
#         np.array([1, 0.6, 0]),
#     ])
#
#     # areaNames = ['allarea', 'small', 'medium', 'large']
#     areaNames = ['allarea']
#     types = ['C50']  # we only need 'C50'
#     catIds = [0, 1, 2, 3, 4, 5, 6, 8, 9, 11]
#     for i in range(len(areaNames)):
#         if class_name.startswith("allclass"):  # Check if class_name is "allclass"
#             # Create a new ps array that only contains the scores for the categories you care about
#             ps_modified = ps[:, :, catIds]
#             # Recalculate area_ps and ps_curve with the modified ps array
#             area_ps = ps_modified[..., i, 0]
#             # C50
#             # aps = [ps_.mean() for ps_ in area_ps[1:2]]
#             # C75
#             # aps = [ps_.mean() for ps_ in area_ps[0:1]]
#             # Loc
#             aps = [ps_.mean() for ps_ in area_ps[2:3]]
#             ps_curve = [area_ps[2].mean(axis=1) if area_ps[2].ndim > 1 else area_ps[2]]
#
#             ax.plot(rs, ps_curve[0], linewidth=3, color='blue', label=f'{class_name}')
#         else:
#             area_ps = ps[..., i, 0]
#             figure_title = iou_type + '-' + class_name + '-' + areaNames[i]
#             # types = ['C75', 'C50', 'Loc', 'Sim', 'Oth', 'BG', 'FN']
#             # C50
#             # aps = [ps_.mean() for ps_ in area_ps[1:2]]  # average precision (ap) for 'C50'#去掉area_ps[1：2]
#             # C75
#             # aps = [ps_.mean() for ps_ in area_ps[0:1]]
#             # Loc
#             aps = [ps_.mean() for ps_ in area_ps[2:3]]
#             # Sim
#             # aps = [ps_.mean() for ps_ in area_ps[3:4]]
#             # Oth
#             # aps = [ps_.mean() for ps_ in area_ps[4:5]]
#             # BG
#             # aps = [ps_.mean() for ps_ in area_ps[5:6]]
#             # # FN
#             # aps = [ps_.mean() for ps_ in area_ps[6:7]]
#             # aps = [ps_.mean() for ps_ in area_ps]  # average precision (ap)
#             # ps_curve = [
#             #     area_ps[i].mean(axis=1) if area_ps[i].ndim > 1 else area_ps[i]
#             #     for i in range(len(types))
#             # ]
#             # # C50 每次需要改这个索引
#             # ps_curve = [
#             #     area_ps[1].mean(axis=1) if area_ps[1].ndim > 1 else area_ps[1]
#             # ]
#             # C75
#             # ps_curve = [area_ps[0].mean(axis=1) if area_ps[0].ndim > 1 else area_ps[0]]
#             # Loc
#             ps_curve = [area_ps[2].mean(axis=1) if area_ps[2].ndim > 1 else area_ps[2]]
#             # ax.plot(rs, ps_curve[0], color=cs[0], label=f'{class_name} {aps[0]:.3f}')  # plot(recall, precision)
#             ax.plot(rs, ps_curve[0], label=f'{class_name} {aps[0]:.3f}')
#         ax.set_xlabel('Recall')
#         ax.set_ylabel('Precision')
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
#         ax.set_title('Precision-Recall Curve', fontsize=12)
#
#     fig = ax.get_figure()
#     fig.savefig(outDir + '/PR_curve_Loc@0.5.png', dpi=300)
#     plt.close(fig)
#     return aps[0]

def makeplot(rs, ps, outDir, class_name, iou_type, ax):
    cs = np.vstack([
        np.array(hex_to_rgb('FFC9C9')),  # light red for C75
        np.array(hex_to_rgb('FF914D')),  # light blue for C50
        np.array([0.31, 0.51, 0.74]),
        np.array([0.75, 0.31, 0.30]),
        np.array([0.36, 0.90, 0.38]),
        np.array([0.50, 0.39, 0.64]),
        np.array([1, 0.6, 0]),
    ])

    areaNames = ['allarea']
    types = ['C50']  # we only need 'C50'
    catIds = [0, 1, 2, 3, 4, 5, 6, 8, 9, 11]
    for i in range(len(areaNames)):
        if class_name.startswith("allclass"):  # Check if class_name is "allclass"
            ps_modified = ps[:, :, catIds]
            area_ps = ps_modified[..., i, 0]
            # C50
            # aps = [ps_.mean() for ps_ in area_ps[1:2]]
            # ps_curve = [area_ps[1].mean(axis=1) if area_ps[1].ndim > 1 else area_ps[1]]
            # # C75
            # aps = [ps_.mean() for ps_ in area_ps[0:1]]
            # ps_curve = [area_ps[0].mean(axis=1) if area_ps[0].ndim > 1 else area_ps[0]]
            # Loc
            aps = [ps_.mean() for ps_ in area_ps[2:3]]
            ps_curve = [area_ps[2].mean(axis=1) if area_ps[2].ndim > 1 else area_ps[2]]
            ax.plot(rs, ps_curve[0], linewidth=3, color='blue', label=f'{class_name}')
        else:
            area_ps = ps[..., i, 0]
            figure_title = iou_type + '-' + class_name + '-' + areaNames[i]
            # C50
            # aps = [ps_.mean() for ps_ in area_ps[1:2]] #C50
            # ps_curve = [area_ps[1].mean(axis=1) if area_ps[1].ndim > 1 else area_ps[1]]
            # C75
            # aps = [ps_.mean() for ps_ in area_ps[0:1]]
            # ps_curve = [area_ps[0].mean(axis=1) if area_ps[0].ndim > 1 else area_ps[0]]
            # Loc
            aps = [ps_.mean() for ps_ in area_ps[2:3]]
            ps_curve = [area_ps[2].mean(axis=1) if area_ps[2].ndim > 1 else area_ps[2]]
            ax.plot(rs, ps_curve[0], label=f'{class_name} {aps[0]:.3f}')

        ax.set_xlabel('Recall', fontsize=14, fontname='Times New Roman')
        ax.set_ylabel('Precision', fontsize=14, fontname='Times New Roman')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(prop={'family': 'Times New Roman', 'size': 12}, bbox_to_anchor=(1.04, 1), loc='upper left')
        ax.set_title('Precision-Recall Curve', fontsize=16, fontname='Times New Roman')

    fig = ax.get_figure()
    fig.savefig(outDir + '/PR_curve_map0.5.png', dpi=300)
    plt.close(fig)
    return aps[0]


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 0 and height <= 1:  # for percent values
            text_label = '{:2.0f}'.format(height * 100)
        else:
            text_label = '{:2.0f}'.format(height)
        ax.annotate(
            text_label,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 1),  # 3 points vertical offset
            textcoords='offset points',
            ha='center',
            va='bottom',
            # fontsize='x-small',
            fontsize=12
        )


def makebarplot(rs, ps, outDir, class_name, iou_type):
    areaNames = ['allarea', 'small', 'medium', 'large']
    # types = ['C75', 'C50', 'Loc', 'Sim', 'Oth', 'BG', 'FN']
    types = ['C75', 'C50', 'Loc']
    fig, ax = plt.subplots()
    x = np.arange(len(areaNames))  # the areaNames locations
    width = 0.60  # the width of the bars
    rects_list = []
    figure_title = iou_type + '-' + class_name + '-' + 'AP' #  Bar Plot

    # for i in range(len(types) - 1):
    for i in range(len(types)):
        type_ps = ps[i, ..., 0]
        aps = [ps_.mean() for ps_ in type_ps.T]
        rects_list.append(
            ax.bar(
                x - width / 2 + (i + 1) * width / len(types),
                aps,
                width / len(types),
                label=types[i],
            ))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Average Precision (mAP)', fontsize=16, fontname='Times New Roman')
    ax.set_title(figure_title, fontsize=18, fontname='Times New Roman')
    ax.set_xticks(x)
    ax.set_xticklabels(areaNames, fontsize=12, fontname='Times New Roman')
    # Add semi-transparent legend
    ax.legend(prop={'family': 'Times New Roman', 'size': 12}, framealpha=0.3)#0.1试一试

    # Add score texts over bars
    for rects in rects_list:
        autolabel(ax, rects)

    # Save plot
    fig.tight_layout()
    fig.savefig(outDir + f'/{figure_title}.png',dpi=300)
    plt.close(fig)



def get_gt_area_group_numbers(cocoEval):
    areaRng = cocoEval.params.areaRng
    areaRngStr = [str(aRng) for aRng in areaRng]
    areaRngLbl = cocoEval.params.areaRngLbl
    areaRngStr2areaRngLbl = dict(zip(areaRngStr, areaRngLbl))
    areaRngLbl2Number = dict.fromkeys(areaRngLbl, 0)
    for evalImg in cocoEval.evalImgs:
        if evalImg:
            for gtIgnore in evalImg['gtIgnore']:
                if not gtIgnore:
                    aRngLbl = areaRngStr2areaRngLbl[str(evalImg['aRng'])]
                    areaRngLbl2Number[aRngLbl] += 1
    return areaRngLbl2Number


def make_gt_area_group_numbers_plot(cocoEval, outDir, verbose=True):
    areaRngLbl2Number = get_gt_area_group_numbers(cocoEval)
    areaRngLbl = areaRngLbl2Number.keys()
    if verbose:
        print('number of annotations per area group:', areaRngLbl2Number)

    # Init figure
    fig, ax = plt.subplots()
    x = np.arange(len(areaRngLbl))  # the areaNames locations
    width = 0.60  # the width of the bars
    figure_title = 'Number of Annotations Per Area Group'

    rects = ax.bar(x, areaRngLbl2Number.values(), width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Number of Annotations', fontsize=16, fontname='Times New Roman')
    ax.set_title(figure_title, fontsize=18, fontname='Times New Roman')
    ax.set_xticks(x)
    ax.set_xticklabels(areaRngLbl, fontname='Times New Roman')
    ax.tick_params(axis='x', labelsize=12)

    # Add score texts over bars
    autolabel(ax, rects)
    # Save plot
    fig.tight_layout()
    fig.savefig(outDir + f'/{figure_title}.png', dpi=300)
    plt.close(fig)


def make_gt_area_histogram_plot(cocoEval, outDir):
    n_bins = 100
    areas = [ann['area'] for ann in cocoEval.cocoGt.anns.values()]

    # init figure
    figure_title = 'GT Annotation Areas Histogram Plot'
    fig, ax = plt.subplots()

    # Set the number of bins
    ax.hist(np.sqrt(areas), bins=n_bins)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Squareroot Area', fontsize=16, fontname='Times New Roman')
    ax.set_ylabel('Number of Annotations', fontsize=16, fontname='Times New Roman')
    ax.set_title(figure_title, fontsize=18, fontname='Times New Roman')

    # Save plot
    fig.tight_layout()
    fig.savefig(outDir + f'/{figure_title}.png', dpi=300)
    plt.close(fig)



def analyze_individual_category(catId,
                                cocoDt,
                                cocoGt,
                                iou_type,
                                areas=None):
    nm = cocoGt.loadCats(catId)[0]
    print(f'--------------analyzing {catId}-{nm["name"]}---------------')
    ps_ = {}
    dt = copy.deepcopy(cocoDt)
    nm = cocoGt.loadCats(catId)[0]
    imgIds = cocoGt.getImgIds()
    dt_anns = dt.dataset['annotations']
    select_dt_anns = []
    for ann in dt_anns:
        if ann['category_id'] == catId:
            select_dt_anns.append(ann)
    dt.dataset['annotations'] = select_dt_anns
    dt.createIndex()
    # compute precision but ignore superclass confusion
    gt = copy.deepcopy(cocoGt)
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [100]
    cocoEval.params.iouThrs = [0.1]
    cocoEval.params.useCats = 1
    if areas:
        cocoEval.params.areaRng = [[0**2, areas[2]], [0**2, areas[0]],
                                   [areas[0], areas[1]], [areas[1], areas[2]]]
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_supercategory = cocoEval.eval['precision'][0, :, catId, :, :]
    ps_['ps_supercategory'] = ps_supercategory
    # compute precision but ignore any class confusion
    gt = copy.deepcopy(cocoGt)
    for idx, ann in enumerate(gt.dataset['annotations']):
        if ann['category_id'] != catId:
            gt.dataset['annotations'][idx]['ignore'] = 1
            gt.dataset['annotations'][idx]['iscrowd'] = 1
            gt.dataset['annotations'][idx]['category_id'] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [100]
    cocoEval.params.iouThrs = [0.1]
    cocoEval.params.useCats = 1
    if areas:
        cocoEval.params.areaRng = [[0**2, areas[2]], [0**2, areas[0]],
                                   [areas[0], areas[1]], [areas[1], areas[2]]]
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_allcategory = cocoEval.eval['precision'][0, :, catId, :, :]
    ps_['ps_allcategory'] = ps_allcategory
    return catId, ps_

def analyze_results(res_file,
                    ann_file,
                    res_types,
                    out_dir,
                    extraplots=None,
                    areas=None):
    for res_type in res_types:
        assert res_type in ['bbox', 'segm']
    if areas:
        assert len(areas) == 3, '3 integers should be specified as areas, \
            representing 3 area regions'

    directory = os.path.dirname(out_dir + '/')
    if not os.path.exists(directory):
        print(f'-------------create {out_dir}-----------------')
        os.makedirs(directory)

    cocoGt = COCO(ann_file)
    cocoDt = cocoGt.loadRes(res_file)
    imgIds = cocoGt.getImgIds()
    for res_type in res_types:
        res_out_dir = out_dir + '/' + res_type + '/'
        res_directory = os.path.dirname(res_out_dir)
        if not os.path.exists(res_directory):
            print(f'-------------create {res_out_dir}-----------------')
            os.makedirs(res_directory)
        iou_type = res_type
        cocoEval = COCOeval(
            copy.deepcopy(cocoGt), copy.deepcopy(cocoDt), iou_type)
        cocoEval.params.imgIds = imgIds
        cocoEval.params.iouThrs = [0.75, 0.5, 0.1]
        cocoEval.params.maxDets = [100]
        if areas:
            cocoEval.params.areaRng = [[0**2, areas[2]], [0**2, areas[0]],
                                       [areas[0], areas[1]],
                                       [areas[1], areas[2]]]
        cocoEval.evaluate()
        cocoEval.accumulate()
        ps = cocoEval.eval['precision']
        ps = np.vstack([ps, np.zeros((4, *ps.shape[1:]))])
        # catIds = cocoGt.getCatIds()
        catIds = [0, 1, 2, 3, 4, 5, 6, 8, 9, 11] # 这里 控制计算并保存的类别
        # catIds = [all_catIds[i] for i in [0, 1, 2, 3, 4, 6, 8, 9, 11]]

        recThrs = cocoEval.params.recThrs
        with Pool(processes=48) as pool:
            args = [(catId, cocoDt, cocoGt, iou_type, areas) for catId in catIds]
            analyze_results = pool.starmap(analyze_individual_category, args)
            # args = [(catId, cocoDt, cocoGt, iou_type, areas) for catId in catIds]
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        total_ap = 0
        for catId in catIds:
            nm = cocoGt.loadCats(catId)[0]
            print(f'--------------saving {catId}-{nm["name"]}---------------')
            analyze_result = next(result for result in analyze_results if result[0] == catId)
            ps_supercategory = analyze_result[1]['ps_supercategory']
            ps_allcategory = analyze_result[1]['ps_allcategory']
            # ... the rest of your loop ...
            # compute precision but ignore superclass confusion
            ps[3, :, catId, :, :] = ps_supercategory
            # compute precision but ignore any class confusion
            ps[4, :, catId, :, :] = ps_allcategory
            # fill in background and false negative errors and plot
            ps[ps == -1] = 0
            ps[5, :, catId, :, :] = ps[4, :, catId, :, :] > 0
            ps[6, :, catId, :, :] = 1.0
            # ap = makeplot(recThrs, ps[:, :, catId], res_out_dir, nm['name'], iou_type,ax)
            # print(f'Before calling makeplot: {nm["name"]}')  # Print class name before calling makeplot
            ap = makeplot(recThrs, ps[:, :, catId], res_out_dir, nm['name'], iou_type, ax)
            # print(f'After calling makeplot: {nm["name"]}')  # Print class name after calling makeplot
            # if "allclass" not in nm["name"]:
            #     total_ap += ap
            total_ap += ap

            if extraplots:
                makebarplot(recThrs, ps[:, :, catId], res_out_dir, nm['name'],
                            iou_type)
        average_ap = total_ap / len(catIds)

        makeplot(recThrs, ps, res_out_dir, f'allclass {average_ap:.3f} mAP@0.5', iou_type, ax)
        # makeplot(recThrs, ps, res_out_dir, f'allclass {average_ap:.3f} mAP@0.75', iou_type, ax)
        # makeplot(recThrs, ps, res_out_dir, f'allclass {average_ap:.3f} Loc@0.5', iou_type, ax)
        # # makeplot(recThrs, ps, res_out_dir, f'allclass {average_ap:.3f} Sim@0.5', iou_type, ax)
        # # makeplot(recThrs, ps, res_out_dir, f'allclass {average_ap:.3f} oth@0.5', iou_type, ax)
        # # makeplot(recThrs, ps, res_out_dir, f'allclass {average_ap:.3f} BG@0.5', iou_type, ax)
        # makeplot(recThrs, ps, res_out_dir, f'allclass {average_ap:.3f} FN@0.5', iou_type, ax)
        if extraplots:
            makebarplot(recThrs, ps, res_out_dir, 'allclass', iou_type)
            make_gt_area_group_numbers_plot(
                cocoEval=cocoEval, outDir=res_out_dir, verbose=True)
            make_gt_area_histogram_plot(cocoEval=cocoEval, outDir=res_out_dir)

"""
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP.bbox.json
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/coco_error_analysis_results
--ann=/root/autodl-tmp/mmdetection/mmdet/data/solar_cell_EL_image_coco_CLAFE/annotations/instances_val2017.json

//Users//wuzhongze//Documents//中南大学科研//2023论文发表//2023论文//2023TII//实验结果记录与对比//多指标可视化同图对比//neck（消融实验）//FPN//V100-32G(以下都是FPN结构)//tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP（62.8）//20230705_092758//vis_data//20230705_092758.json
//Users//wuzhongze//Documents//中南大学科研//2023论文发表//2023论文//2023TII//实验结果记录与对比//多指标可视化同图对比//neck（消融实验）//FPN//V100-32G(以下都是FPN结构)//tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP（62.8）//coco_error_analysis_results-4
--ann=/Users/wuzhongze/Documents/中南大学科研/2023论文发表/2023论文/2023TII/数据集/solar_cell_EL_image_coco_CLAFE/annotations/instances_val2017.json


//Users//wuzhongze//Documents//中南大学科研//2023论文发表//2023论文//2023TII//实验结果记录与对比//多指标可视化同图对比//neck（消融实验）//FPN//V100-32G(以下都是FPN结构)//tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP（62.8）//20230705_092758//vis_data//20230705_092758.json
//Users//wuzhongze//Documents//中南大学科研//2023论文发表//2023论文//2023TII//实验结果记录与对比//多指标可视化同图对比//neck（消融实验）//FPN//V100-32G(以下都是FPN结构)//tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP（62.8）//coco_error_analysis_results-4
--ann=/Users/wuzhongze/Documents/中南大学科研/2023论文发表/2023论文/2023TII/数据集/solar_cell_EL_image_coco_CLAFE/annotations/instances_val2017.json

/root/mmdetection-3.x/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/20230613_102734/vis_data/20230613_102734.json
/root/mmdetection-3.x/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/coco_error_analysis_results
--ann=/root/autodl-tmp/solar_cell_EL_image_coco/annotations/instances_val2017.json

"""
def main():
    parser = ArgumentParser(description='COCO Error Analysis Tool')
    parser.add_argument('result', help='result file (json format) path')
    parser.add_argument('out_dir', help='dir to save analyze result images')
    parser.add_argument(
        '--ann',
        default='data/coco/annotations/instances_val2017.json',
        help='annotation file path')
    parser.add_argument(
        '--types', type=str, nargs='+', default=['bbox'], help='result types')
    parser.add_argument(
        '--extraplots',
        action='store_true',
        help='export extra bar/stat plots')
    parser.add_argument(
        '--areas',
        type=int,
        nargs='+',
        default=[1024, 9216, 10000000000],
        help='area regions')
    args = parser.parse_args()
    analyze_results(
        args.result,
        args.ann,
        args.types,
        out_dir=args.out_dir,
        extraplots=True,
        areas=args.areas)


if __name__ == '__main__':
    main()
