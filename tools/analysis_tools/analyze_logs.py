# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def cal_train_time(log_dicts, args):
    for i, log_dict in enumerate(log_dicts):
        print(f'{"-" * 5}Analyze train time of {args.json_logs[i]}{"-" * 5}')
        all_times = []
        for epoch in log_dict.keys():
            if args.include_outliers:
                all_times.append(log_dict[epoch]['time'])
            else:
                all_times.append(log_dict[epoch]['time'][1:])
        if not all_times:
            raise KeyError(
                'Please reduce the log interval in the config so that'
                'interval is less than iterations of one epoch.')
        all_times = np.array(all_times)
        epoch_ave_time = all_times.mean(-1)
        slowest_epoch = epoch_ave_time.argmax()
        fastest_epoch = epoch_ave_time.argmin()
        std_over_epoch = epoch_ave_time.std()
        print(f'slowest epoch {slowest_epoch + 1}, '
              f'average time is {epoch_ave_time[slowest_epoch]:.4f}')
        print(f'fastest epoch {fastest_epoch + 1}, '
              f'average time is {epoch_ave_time[fastest_epoch]:.4f}')
        print(f'time std over epochs is {std_over_epoch:.4f}')
        print('average iter time: {:.4f} s/iter'.format(float(np.mean(np.sum(all_times)))))
        #print(f'average iter time: {np.mean(all_times):.4f} s/iter')
        print()

#
# def plot_curve(log_dicts, args):
#     if args.backend is not None:
#         plt.switch_backend(args.backend)
#     sns.set_style(args.style)
#     # if legend is None, use {filename}_{key} as legend
#     legend = args.legend
#     if legend is None:
#         legend = []
#         for json_log in args.json_logs:
#             for metric in args.keys:
#                 legend.append(f'{json_log}_{metric}')
#     assert len(legend) == (len(args.json_logs) * len(args.keys))
#     metrics = args.keys
#
#     num_metrics = len(metrics)
#     for i, log_dict in enumerate(log_dicts):
#         epochs = list(log_dict.keys())
#         for j, metric in enumerate(metrics):
#             print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
#             if metric not in log_dict[epochs[int(args.eval_interval) - 1]]:
#                 if 'mAP' in metric:
#                     raise KeyError(
#                         f'{args.json_logs[i]} does not contain metric '
#                         f'{metric}. Please check if "--no-validate" is '
#                         'specified when you trained the model.')
#                 raise KeyError(
#                     f'{args.json_logs[i]} does not contain metric {metric}. '
#                     'Please reduce the log interval in the config so that '
#                     'interval is less than iterations of one epoch.')
#
#             if 'mAP' in metric:
#                 xs = []
#                 ys = []
#                 for epoch in epochs:
#                     ys += log_dict[epoch][metric]
#                     xs += [epoch]
#                 plt.xlabel('epoch')
#                 plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
#             else:
#                 xs = []
#                 ys = []
#                 for epoch in epochs:
#                     iters = log_dict[epoch]['step']
#                     xs.append(np.array(iters))
#                     ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
#                 xs = np.concatenate(xs)
#                 ys = np.concatenate(ys)
#                 plt.xlabel('iter')
#                 plt.plot(
#                     xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
#             plt.legend()
#         if args.title is not None:
#             plt.title(args.title)
#     if args.out is None:
#         plt.show()
#     else:
#         print(f'save curve to: {args.out}')
#         plt.savefig(args.out)
#         plt.cla()

def plot_curve(log_dicts, args):
    if args.backend is not None:
        plt.switch_backend(args.backend)
    sns.set_style(args.style)
    # if legend is None, use {filename}_{key} as legend
    legend = args.legend
    if legend is None:
        legend = []
        for json_log in args.json_logs:
            for metric in args.keys:
                legend.append(f'{json_log}_{metric}')
    assert len(legend) == (len(args.json_logs) * len(args.keys))
    metrics = args.keys

    num_metrics = len(metrics)
    fig, ax = plt.subplots(dpi=300)
    for i, log_dict in enumerate(log_dicts):
        epochs = list(log_dict.keys())
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')
            if metric not in log_dict[epochs[int(args.eval_interval) - 1]]:
                if 'mAP' in metric:
                    raise KeyError(
                        f'{args.json_logs[i]} does not contain metric '
                        f'{metric}. Please check if "--no-validate" is '
                        'specified when you trained the model.')
                raise KeyError(
                    f'{args.json_logs[i]} does not contain metric {metric}. '
                    'Please reduce the log interval in the config so that '
                    'interval is less than iterations of one epoch.')

            if 'mAP' in metric:
                xs = []
                ys = []
                for epoch in epochs:
                    ys += log_dict[epoch][metric]
                    xs += [epoch]
                ax.set_xlabel('epoch', fontsize=16, fontname='Times New Roman')
                ax.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
                ax.set_ylabel('mAP', fontsize=16, fontname='Times New Roman')
            else:
                xs = []
                ys = []
                for epoch in epochs:
                    iters = log_dict[epoch]['step']
                    xs.append(np.array(iters))
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                ax.set_xlabel('iter', fontsize=16, fontname='Times New Roman')
                ax.plot(xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
                ax.set_ylabel('Loss', fontsize=16, fontname='Times New Roman')
            ax.legend(prop={'family':'Times New Roman', 'size':12})
        if args.title is not None:
            ax.set_title(args.title, fontsize=18, fontname='Times New Roman')
    if args.out is None:
        plt.show()
    else:
        print(f'save curve to: {args.out}')
        plt.savefig(args.out, dpi=300)
        plt.cla()





def add_plot_parser(subparsers):
    parser_plt = subparsers.add_parser(
        'plot_curve', help='parser for plotting curves')
    parser_plt.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_plt.add_argument(
        '--keys',
        type=str,
        nargs='+',
        default=['bbox_mAP'],
        help='the metric that you want to plot')
    parser_plt.add_argument(
        '--start-epoch',
        type=str,
        default='1',
        help='the epoch that you want to start')
    parser_plt.add_argument(
        '--eval-interval',
        type=str,
        default='1',
        help='the eval interval when training')
    parser_plt.add_argument('--title', type=str, help='title of figure')
    parser_plt.add_argument(
        '--legend',
        type=str,
        nargs='+',
        default=None,
        help='legend of each plot')
    parser_plt.add_argument(
        '--backend', type=str, default=None, help='backend of plt')
    parser_plt.add_argument(
        '--style', type=str, default='dark', help='style of plt')
    parser_plt.add_argument('--out', type=str, default=None)


def add_time_parser(subparsers):
    parser_time = subparsers.add_parser(
        'cal_train_time',
        help='parser for computing the average time per training iteration')
    parser_time.add_argument(
        'json_logs',
        type=str,
        nargs='+',
        help='path of train log in json format')
    parser_time.add_argument(
        '--include-outliers',
        action='store_true',
        help='include the first value of every epoch when computing '
        'the average time')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Json Log')
    # currently only support plot curve and calculate average train time
    subparsers = parser.add_subparsers(dest='task', help='task parser')
    add_plot_parser(subparsers)
    add_time_parser(subparsers)
    args = parser.parse_args()
    return args


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            epoch = 1
            for i, line in enumerate(log_file):
                log = json.loads(line.strip())
                val_flag = False
                # skip lines only contains one key
                if not len(log) > 1:
                    continue

                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)

                # for k, v in log.items():
                #     if '/' in k:
                #         log_dict[epoch][k.split('/')[-1]].append(v)
                #         val_flag = True
                #     elif val_flag:
                #         continue
                #     else:
                #         log_dict[epoch][k].append(v)
                for k, v in log.items():
                    if '/' in k:
                        log_dict[epoch][k].append(v)
                        val_flag = True
                    elif val_flag:
                        continue
                    else:
                        log_dict[epoch][k].append(v)

                if 'epoch' in log.keys():
                    epoch = log['epoch']

    return log_dicts


def main():
    args = parse_args()

    json_logs = args.json_logs
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    eval(args.task)(log_dicts, args)
"""
pip install seaborn
plot_curve /hy-tmp/mmdetection/tools/work_dirs/yolox_s_8xb8-300e_coco_solar_cell/20230302_172211/vis_data/20230302_172211.json --keys loss_cls loss_bbox --legend loss_cls loss_bbox
--out /hy-tmp/mmdetection/tools/work_dirs/yolox_s_8xb8-300e_coco_solar_cell/yolox_s_8xb8-300e_coco_solar_cell_loss300.pdf

plot_curve /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell/20230302_220227/vis_data/20230302_220227.json --keys loss_cls loss_bbox loss --legend loss_cls loss_bbox loss
--out /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell_loss300.pdf
可以发现loss还在有角度下降还能继续跑

plot_curve /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell/20230302_220227/vis_data/20230302_220227.json --keys bbox_mAP bbox_mAP_50 bbox_mAP_75 --legend bbox_mAP bbox_mAP_50 bbox_mAP_75
--out /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell_map.pdf
可以发现map值还在有角度下降还能继续跑

plot_curve /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_nas-fpn_ms-2x_coco_solar_cell/20230310_112523/vis_data/20230310_112523.json --keys loss_cls loss_bbox loss --legend loss_cls loss_bbox loss
--out /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_nas-fpn_ms-2x_coco_solar_cell/tood_swin-s-p4-w12_nas-fpn_ms-2x_coco_solar_cell_loss24.pdf
可以发现loss还在有角度下降还能继续跑

plot_curve /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_new_solar_data/20230314_101732_24_epoch/vis_data/20230314_101732.json --keys loss_cls loss_bbox loss --legend loss_cls loss_bbox loss
--out /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_new_solar_data/tood_swin-s-p4-w12_fpn_ms-2x_coco_new_solar_data_loss24.pdf
可以发现loss还在有角度下降还能继续跑

plot_curve /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_new_solar_data/20230314_101732_24_epoch/vis_data/20230314_101732.json --keys bbox_mAP bbox_mAP_50 bbox_mAP_75 --legend bbox_mAP bbox_mAP_50 bbox_mAP_75
--out /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_new_solar_data/tood_swin-s-p4-w12_fpn_ms-2x_coco_new_solar_data_map24.pdf
可以发现map值还在有角度下降还能继续跑


plot_curve /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone/20230317_092455/vis_data/20230317_092455.json --keys loss_cls loss_bbox loss --legend loss_cls loss_bbox loss
--out /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone_loss23.pdf
可以发现map值还在有角度下降还能继续跑

plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_EL-image-aug/20230412_080622/vis_data/20230412_080622.json --keys loss_cls loss_bbox loss --legend loss_cls loss_bbox loss
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_EL-image-aug/tood_swin-s-p4-w12_fpn_ms-2x_coco_EL-image-aug_loss21.pdf

plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin/20230424_130925/vis_data/20230424_130925.json  --keys loss_cls loss_bbox loss --legend loss_cls loss_bbox loss
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin_loss21.pdf

plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/20230612_194225/vis_data/20230612_194225.json  --keys loss_cls loss_bbox loss --legend loss_cls loss_bbox loss
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin_loss24.pdf

plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-HCBP/20230612_194601/vis_data/20230612_194601.json --keys loss --legend loss
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-HCBP/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-HCBP-24.pdf

# 比较两次不同方法的loss指标
plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/20230612_194225/vis_data/20230612_194225.json /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-HCBP/20230612_194601/vis_data/20230612_194601.json --keys loss_cls loss_bbox loss --legend base_cls base_bbox base HCBP_cls HCBP_bbox HCBP
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/loss_all_compare_base_HCBP.pdf

# 比较两次不同方法的map指标
plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/20230612_194225/vis_data/20230612_194225.json /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-HCBP/20230612_194601/vis_data/20230612_194601.json --keys "coco/bbox_mAP"--legend base HCBP --title bbox_mAP
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/bbox_mAP_compare_base_HCBP.pdf

# 比较两次不同方法的map/map50/map75指标
plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/20230612_194225/vis_data/20230612_194225.json /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-HCBP/20230612_194601/vis_data/20230612_194601.json --keys coco/bbox_mAP coco/bbox_mAP_50 coco/bbox_mAP_75 --legend base base_50 base_75 HCBP HCBP_50 HCBP_75 --title bbox_mAP/50/75
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/bbox_mAP_all_compare_base_HCBP.pdf
# 比较两次不同方法的maps/mapm/mapl指标
plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/20230612_194225/vis_data/20230612_194225.json 
/root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-HCBP/20230612_194601/vis_data/20230612_194601.json 
--keys coco/bbox_mAP_s coco/bbox_mAP_m coco/bbox_mAP_l --legend base_s base_m base_l HCBP_s HCBP_m HCBP_l --title bbox_mAP_s/m/l
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/bbox_mAP_s_m_l_compare_base_HCBP.pdf

# 比较两次不同方法的时间指标
plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/20230612_194225/vis_data/20230612_194225.json /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-HCBP/20230612_194601/vis_data/20230612_194601.json --keys loss_cls loss_bbox loss --legend base_cls base_bbox base HCBP_cls HCBP_bbox HCBP
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/loss_all_compare_base_HCBP.pdf

# 比较两次不同方法的map指标
plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/20230612_194225/vis_data/20230612_194225.json /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-HCBP/20230612_194601/vis_data/20230612_194601.json --keys "coco/bbox_mAP"--legend base HCBP --title bbox_mAP
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin/bbox_mAP_compare_base_HCBP.pdf
"coco/bbox_mAP": 0.258, "coco/bbox_mAP_50": 0.441, "coco/bbox_mAP_75": 0.226, "coco/bbox_mAP_s": 0.036, "coco/bbox_mAP_m": 0.188, "coco/bbox_mAP_l": 0.266,

python tools/analysis_tools/analyze_logs.py plot_curve /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-acmix/20230329_105438-24/vis_data/20230329_105438.json --keys loss_cls loss_bbox loss --legend loss_cls loss_bbox loss --out /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-acmix/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-acmix-loss24.pdf
# 计算速度
python tools/analyze_logs.py cal_train_time /hy-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_solar_cell/20230302_220227/vis_data/20230302_220227.json --include-outliers
python tools/analyze_logs.py cal_train_time /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_multi_backbone-EL-image-apn-data3-process-orin/20230424_130925/vis_data/20230424_130925.json --include-outliers

# 最新可视化0707：
plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-HCBP/20230612_194601/vis_data/20230612_194601.json --keys loss --legend loss
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-HCBP/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-HCBP-24.pdf

plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/20230705_092758/vis_data/20230705_092758.json --keys loss --legend loss
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_BEACON-24.pdf



plot_curve /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/20230705_092758/vis_data/20230705_092758.json --keys loss --legend loss
--out /root/autodl-tmp/mmdetection/tools/work_dirs/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_BEACON-24.pdf


plot_curve
/Users/wuzhongze/Documents/中南大学科研/2023论文发表/2023论文/2023TII/实验结果记录与对比/多指标可视化同图对比/neck（消融实验）/FPN/V100-32G(以下都是FPN结构)/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP（62.8）/20230705_092758/vis_data/20230705_092758.json
--keys
loss_cls
loss_bbox
loss
--legend
loss_cls
loss_bbox
loss
--out
/Users/wuzhongze/Documents/中南大学科研/2023论文发表/2023论文/2023TII/实验结果记录与对比/多指标可视化同图对比/neck（消融实验）/FPN/V100-32G(以下都是FPN结构)/tood_swin-s-p4-w12_fpn_ms-2x_coco_-EL-image-orin-FPN_acris_APN_CLAFE-HCBP（62.8）/20230705_092758/vis_data/BEACON_loss.png
--title
"Training Loss for using QualityFocalLoss and GIoULoss"


plot_curve
/root/ConsistentTeacher-main/work_dirs/consistent_teacher_r50_fpn_voc0924_72k/20230924_111016.log.json

plot_curve
/root/ConsistentTeacher-main/work_dirs/consistent_teacher_r50_fpn_voc0924_72k_mha_fam3d/20230926_230506.log.json
--keys
teacher.bbox_mAP
student.bbox_mAP
--legend
mha_fam3d_teacher
mha_fam3d_student
--title bbox_mAP_s/t
--out /Users/wuzhongze/Documents/中南大学科研/2023论文发表/2023论文/2023半监督期刊/实验/数据可视化/mha_fam3d_map.pdf
"""


if __name__ == '__main__':
    main()
