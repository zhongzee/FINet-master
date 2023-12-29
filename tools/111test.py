import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_and_save(img, title, output_folder):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.savefig(os.path.join(output_folder, f"{title}.png"))
    plt.show()

def preprocess_image(image_path, output_folder):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 灰度图像
    visualize_and_save(img, '01_gray', output_folder)
    # 中值滤波
    median_filtered = cv2.medianBlur(img, 5)
    visualize_and_save(median_filtered, '02_median_filtered', output_folder)
    # 顶帽变换
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(median_filtered, cv2.MORPH_TOPHAT, kernel)
    visualize_and_save(tophat, '03_tophat', output_folder)
    # 腐蚀
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(tophat, kernel_erode)
    visualize_and_save(eroded, '04_eroded', output_folder)
    # 膨胀
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(eroded, kernel_dilate)
    visualize_and_save(dilated, '05_dilated', output_folder)
    # 图像分割（Otsu阈值分割）
    ret, segmented = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    visualize_and_save(segmented, '06_segmented', output_folder)

image_path = "//Users/wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar_cell_EL_image_coco//train2017//img000002.jpg"
output_folder = "//Users/wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar_cell_EL_image_coco_process//"
preprocess_image(image_path, output_folder)