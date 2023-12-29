import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set default font to Times New Roman
rcParams['font.family'] = 'Times New Roman'

def ahee(img, clip_limit=2.0, tile_size=(8, 8)):
    # 将图像分成许多小块
    height, width = img.shape
    x_steps = width // tile_size[0]
    y_steps = height // tile_size[1]
    tiles = []
    for y in range(y_steps):
        for x in range(x_steps):
            tile = img[y*tile_size[1]:(y+1)*tile_size[1], x*tile_size[0]:(x+1)*tile_size[0]]
            tiles.append(tile)
    # 对每个小块进行直方图均衡化
    for i in range(len(tiles)):
        tiles[i] = cv2.equalizeHist(tiles[i])
    # 将小块拼合成整个图像
    result = np.zeros((height, width), dtype=np.uint8)
    i = 0
    for y in range(y_steps):
        for x in range(x_steps):
            result[y*tile_size[1]:(y+1)*tile_size[1], x*tile_size[0]:(x+1)*tile_size[0]] = tiles[i]
            i += 1
    # 对整个图像进行 CLAHE 处理
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    result = clahe.apply(result)
    return result

def visualize_and_save(img, title,output_folder):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.savefig(os.path.join(output_folder, f"{title}.png"))
    plt.show()

def bhe(img, threshold=128):
    # 将图像分为低灰度级和高灰度级两部分
    low = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    high = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    low[img < threshold] = img[img < threshold]
    high[img >= threshold] = img[img >= threshold]
    # 对低灰度级和高灰度级分别进行直方图均衡化
    low_eq = cv2.equalizeHist(low)
    high_eq = cv2.equalizeHist(high)
    # 合并低灰度级和高灰度级
    result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    result[img < threshold] = low_eq[img < threshold]
    result[img >= threshold] = high_eq[img >= threshold]
    return result


def find_bounds(integral_proj, threshold):
    start = 0
    end = len(integral_proj) - 1

    for i, val in enumerate(integral_proj):
        if val > threshold:
            start = i
            break

    for i, val in reversed(list(enumerate(integral_proj))):
        if val > threshold:
            end = i
            break

    return start, end

def crop_solar_cell_image(image, threshold_h, threshold_v):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image

    rows, cols = img_gray.shape
    integral_proj_h = np.sum(img_gray, axis=1)
    integral_proj_v = np.sum(img_gray, axis=0)

    top, bottom = find_bounds(integral_proj_h, threshold_h)
    left, right = find_bounds(integral_proj_v, threshold_v)

    cropped_img = image[top:bottom+1, left:right+1]
    return cropped_img



def crop_image(img, threshold):
    # 计算水平和垂直方向的积分投影
    ih = np.sum(img, axis=1)
    jv = np.sum(img, axis=0)

    # 确定太阳能电池片上下边界位置
    h_threshold = np.min(ih) + threshold
    top, bottom = 0, img.shape[0] - 1
    while ih[top] < h_threshold:
        top += 1
    while ih[bottom] < h_threshold:
        bottom -= 1

    # 确定太阳能电池片左右边界位置
    v_threshold = np.max(jv) - threshold
    left, right = 0, img.shape[1] - 1
    while jv[left] < v_threshold:
        left += 1
    while jv[right] < v_threshold:
        right -= 1

    # 裁剪图像
    cropped = img[top:bottom+1, left:right+1]
    return cropped

def rotate_image_to_horizontal(image, edge_threshold=100, line_threshold=150):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image

    edges = cv2.Canny(img_gray, edge_threshold, edge_threshold * 3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, line_threshold)

    max_length = 0
    max_angle = 0
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length > max_length:
            max_length = length
            max_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    if max_angle < -45:
        max_angle += 90
    elif max_angle > 45:
        max_angle -= 90

    rotated_image = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), max_angle, 1)
    result = cv2.warpAffine(image, rotated_image, (image.shape[1], image.shape[0]))

    return result


def dynamic_crop_solar_cell_image(image):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image

    # 计算水平和垂直积分投影
    horizontal_projection = np.sum(img_gray, axis=0)
    vertical_projection = np.sum(img_gray, axis=1)

    # 计算水平和垂直
    threshold_h = np.percentile(horizontal_projection, 5)
    threshold_v = np.percentile(vertical_projection, 5)

    # 寻找太阳能电池片的边界
    left_boundary = 0
    right_boundary = img_gray.shape[1] - 1
    top_boundary = 0
    bottom_boundary = img_gray.shape[0] - 1

    for i in range(img_gray.shape[1]):
        if horizontal_projection[i] > threshold_h:
            left_boundary = i
            break

    for i in range(img_gray.shape[1] - 1, -1, -1):
        if horizontal_projection[i] > threshold_h:
            right_boundary = i
            break

    for i in range(img_gray.shape[0]):
        if vertical_projection[i] > threshold_v:
            top_boundary = i
            break

    for i in range(img_gray.shape[0] - 1, -1, -1):
        if vertical_projection[i] > threshold_v:
            bottom_boundary = i
            break

    # 根据边界裁剪图像
    cropped_image = image[top_boundary:bottom_boundary+1, left_boundary:right_boundary+1]

    return cropped_image

def combined_crop_solar_cell_image(image, alpha_h=0.1, alpha_v=0.1, manual_threshold_h=3500, manual_threshold_v=3500):
    if len(image.shape) == 3:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = image

    # 计算水平和垂直积分投影
    horizontal_projection = np.sum(img_gray, axis=0)
    vertical_projection = np.sum(img_gray, axis=1)

    # 计算水平和垂直阈值
    adaptive_threshold_h = np.percentile(horizontal_projection, 5)
    adaptive_threshold_v = np.percentile(vertical_projection, 5)

    # 结合手动设置阈值和自适应阈值
    threshold_h = alpha_h * adaptive_threshold_h + (1 - alpha_h) * manual_threshold_h
    threshold_v = alpha_v * adaptive_threshold_v + (1 - alpha_v) * manual_threshold_v

    # 寻找太阳能电池片的边界
    left_boundary = 0
    right_boundary = img_gray.shape[1] - 1
    top_boundary = 0
    bottom_boundary = img_gray.shape[0] - 1

    for i in range(img_gray.shape[1]):
        if horizontal_projection[i] > threshold_h:
            left_boundary = i
            break

    for i in range(img_gray.shape[1] - 1, -1, -1):
        if horizontal_projection[i] > threshold_h:
            right_boundary = i
            break

    for i in range(img_gray.shape[0]):
        if vertical_projection[i] > threshold_v:
            top_boundary = i
            break

    for i in range(img_gray.shape[0] - 1, -1, -1):
        if vertical_projection[i] > threshold_v:
            bottom_boundary = i
            break

    # 根据边界裁剪图像
    cropped_image = image[top_boundary:bottom_boundary+1, left_boundary:right_boundary+1]

    return cropped_image



def preprocess_image(image_path, output_folder):
    # 读取图像
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_name_prefix = os.path.splitext(os.path.basename(image_path))[0]
    threshold_h = 1600
    threshold_v = 2000
    # # 图像旋转
    # rotate_image= rotate_image_to_horizontal(img, edge_threshold=100, line_threshold=150)
    # visualize_and_save(rotate_image, f'{image_name_prefix}_01_rotate', output_folder)
    # # 图像裁剪
    # cropped_image = crop_solar_cell_image(rotate_image, threshold_h, threshold_v)
    # # 图像裁剪
    # # cropped = crop_image(img, threshold=10)
    # visualize_and_save(cropped_image, f'{image_name_prefix}_02_cropped', output_folder)
    # # 自适应图像裁剪
    # dynamic_crop_image = dynamic_crop_solar_cell_image(rotate_image)
    # visualize_and_save(dynamic_crop_image, f'{image_name_prefix}_03_dynamic_cropped', output_folder)
    # # 自适应和手动图像裁剪
    # combined_crop_image = combined_crop_solar_cell_image(rotate_image, alpha_h=0.1, alpha_v=0.1, manual_threshold_h=10000, manual_threshold_v=10000)
    # visualize_and_save(combined_crop_image, f'{image_name_prefix}_04_combined_crop_image', output_folder)
    # 灰度图像
    # visualize_and_save(combined_crop_image, f'{image_name_prefix}_03_gray', output_folder)
    # 自适应直方图均衡化(这个最好)

    visualize_and_save(img, f'{image_name_prefix}_orin', output_folder)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))#2，（8，8）
    res_clahe = clahe.apply(img)
    visualize_and_save(res_clahe, f'{image_name_prefix}_04_clahe_1.5_8', output_folder)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))  # 2，（8，8）
    res_clahe2 = clahe.apply(img)
    visualize_and_save(res_clahe2, f'{image_name_prefix}_04_clahe_2_8', output_folder)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))  # 2，（8，8）
    res_clahe3 = clahe.apply(img)
    visualize_and_save(res_clahe3, f'{image_name_prefix}_04_clahe_2.5_8', output_folder)

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))  # 2，（8，8）
    res_clahe3 = clahe.apply(img)
    visualize_and_save(res_clahe3, f'{image_name_prefix}_04_clahe_3_8', output_folder)


    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))#2，（8，8）
    res_clahe4 = clahe.apply(img)
    visualize_and_save(res_clahe4, f'{image_name_prefix}_04_clahe_1.5_4', output_folder)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))  # 2，（8，8）
    res_clahe5 = clahe.apply(img)
    visualize_and_save(res_clahe5, f'{image_name_prefix}_04_clahe_2_4', output_folder)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))  # 2，（8，8）
    res_clahe6 = clahe.apply(img)
    visualize_and_save(res_clahe6, f'{image_name_prefix}_04_clahe_2.5_4', output_folder)

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4, 4))  # 2，（8，8）
    res_clahe7 = clahe.apply(img)
    visualize_and_save(res_clahe7, f'{image_name_prefix}_04_clahe_3_4', output_folder)
    # 双波滤波
    bilateral_filtered = cv2.bilateralFilter(res_clahe7, d=9, sigmaColor=75, sigmaSpace=75)
    visualize_and_save(bilateral_filtered, f'{image_name_prefix}_04_bilateral_filtered_3_4', output_folder)
    # # ahe渐进式均衡化
    # ahe = ahee(img, clip_limit=1.0, tile_size=(108, 108))
    # visualize_and_save(ahe, f'{image_name_prefix}_05_ahe', output_folder)
    # # # 双直方图均衡化
    # ble = bhe(img,threshold=32)
    # visualize_and_save(ble, f'{image_name_prefix}_06_ble', output_folder)
    # # # 直方图均衡化
    # equalized = cv2.equalizeHist(img)
    # visualize_and_save(equalized, f'{image_name_prefix}_07_equalized', output_folder)
    # 中值滤波
    median_filtered = cv2.medianBlur(res_clahe, 5)
    visualize_and_save(median_filtered, f'{image_name_prefix}_08_equalized_median_filtered', output_folder)

    median_filtered = cv2.medianBlur(res_clahe7, 5)
    visualize_and_save(median_filtered, f'{image_name_prefix}_08_equalized_median_filtered_3_4', output_folder)

    # # 顶帽变换
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # tophat = cv2.morphologyEx(median_filtered, cv2.MORPH_TOPHAT, kernel)
    # visualize_and_save(tophat, f'{image_name_prefix}_05_tophat', output_folder)
    # # 腐蚀
    # kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # eroded = cv2.erode(tophat, kernel_erode)
    # visualize_and_save(eroded, f'{image_name_prefix}_04_eroded', output_folder)
    # # 膨胀
    # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # dilated = cv2.dilate(eroded, kernel_dilate)
    # visualize_and_save(dilated, f'{image_name_prefix}_05_dilated', output_folder)
    # # 图像分割（Otsu阈值分割）
    # ret, segmented = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # visualize_and_save(segmented, f'{image_name_prefix}_06_segmented', output_folder)

def visualize_and_save(img, title,output_folder):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.savefig(os.path.join(output_folder, f"{title}.png"))
    plt.show()


def preprocess_images(input_folder, output_folder):
    # 遍历指定文件夹中的所有图像
    for image_file in os.listdir(input_folder):
        # 跳过非图像文件
        if not image_file.lower().endswith('.jpg') and not image_file.lower().endswith('.png'):
            continue

        # 读取图像文件
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_name_prefix = os.path.splitext(image_file)[0]

        # 如果输出文件夹不存在，则创建
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 对图像应用自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        res_clahe = clahe.apply(img)

        # 对图像应用中值滤波
        median_filtered = cv2.medianBlur(res_clahe, 5)
        output_path = os.path.join(output_folder, f"{image_name_prefix}.jpg")
        cv2.imwrite(output_path, median_filtered)
        # plt.savefig(os.path.join(output_folder, f"{image_name_prefix}.jpg"))
        # visualize_and_save(median_filtered, f'{image_name_prefix}', output_folder)

def apply_lowpass_filter(image, cutoff=100):
    # 使用傅立叶变换将图像从空间域转换到频域
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # 创建低通滤波器
    rows, cols = image.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1

    # 应用滤波器并逆傅立叶变换回到空间域
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    # 将图像正规化到0-255的范围并转换为8-bit整数
    img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return img_back

def visualize_hist(image, title, output_folder, image_name_prefix):
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Calculate brightness and contrast
    brightness = np.mean(image)
    contrast = np.std(image)

    # Set up the figure
    fig, ax = plt.subplots()

    # Plot the histogram
    ax.plot(hist, label='Histogram', color='b')
    ax.axvline(brightness, color='r', linestyle='dashed', linewidth=2, label=f'Brightness: {brightness:.2f}')
    ax.axvline(brightness + contrast, color='g', linestyle='dashed', linewidth=2, label=f'Contrast: {contrast:.2f}')
    ax.legend(fontsize=16)
    # ax.set_title(title, fontsize=18, fontname='Times New Roman')

    # Set X and Y labels
    ax.set_xlabel('Pixel Intensity', fontsize=18)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.set_ylim([0, 17500])

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Save the image as PDF
    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, f"{image_name_prefix}_{title.replace(' ', '_')}.pdf"), dpi=600)
    plt.close(fig)




def vis_preprocess_images(input_folder, output_folder):
    # 遍历指定文件夹中的所有图像
    for image_file in os.listdir(input_folder):
        # 跳过非图像文件
        if not image_file.lower().endswith('.jpg') and not image_file.lower().endswith('.png'):
            continue

        # 读取图像文件
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_name_prefix = os.path.splitext(image_file)[0]

        # 如果输出文件夹不存在，则创建
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 对图像应用自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        res_clahe = clahe.apply(img)
        output_path = os.path.join(output_folder, f"{image_name_prefix}_clahe.jpg")
        cv2.imwrite(output_path, res_clahe)

        # 降低亮度
        # 降低亮度
        lowered_brightness = cv2.convertScaleAbs(res_clahe, alpha=0.9, beta=0)
        output_path = os.path.join(output_folder, f"{image_name_prefix}_lowered_brightness.jpg")
        cv2.imwrite(output_path, lowered_brightness)

        # 对图像应用中值滤波
        median_filtered = cv2.medianBlur(res_clahe, 5)
        output_path = os.path.join(output_folder, f"{image_name_prefix}_CLAFE.jpg")
        cv2.imwrite(output_path, median_filtered)

        # # 对图像应用频域低通滤波
        # lowpass_filtered = apply_lowpass_filter(median_filtered)
        # output_path = os.path.join(output_folder, f"{image_name_prefix}_lowpass.jpg")
        # cv2.imwrite(output_path, lowpass_filtered)

        # 可视化直方图，亮度，对比度并保存
        visualize_hist(img, 'Original Image Histogram', output_folder, image_name_prefix)
        visualize_hist(res_clahe, 'Histogram after CLAHE', output_folder, image_name_prefix)
        visualize_hist(lowered_brightness, 'Histogram after Lowering Brightness', output_folder, image_name_prefix)
        visualize_hist(median_filtered, 'Histogram after CLAFE', output_folder, image_name_prefix)


# def enhance_defects_only(img):
#     # 使用Otsu's方法进行图像分割
#     _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # 形态学操作 (Optional)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     # 仅对估计出的缺陷区域应用CLAHE
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
#     clahe_img = clahe.apply(img)
#     enhanced_defects = cv2.bitwise_and(clahe_img, mask)
#
#     # 与原始图像合并
#     inverse_mask = cv2.bitwise_not(mask)
#     non_enhanced_background = cv2.bitwise_and(img, inverse_mask)
#     result = cv2.add(enhanced_defects, non_enhanced_background)
#
#     return result
#
#
# def vis_preprocess_images(input_folder, output_folder):
#     # 遍历指定文件夹中的所有图像
#     for image_file in os.listdir(input_folder):
#         # 跳过非图像文件
#         if not image_file.lower().endswith('.jpg') and not image_file.lower().endswith('.png'):
#             continue
#
#         # 读取图像文件
#         image_path = os.path.join(input_folder, image_file)
#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         image_name_prefix = os.path.splitext(image_file)[0]
#
#         # 如果输出文件夹不存在，则创建
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
#
#         # 使用自定义的函数增强缺陷区域
#         defect_enhanced = enhance_defects_only(img)
#         output_path = os.path.join(output_folder, f"{image_name_prefix}_defect_enhanced.jpg")
#         cv2.imwrite(output_path, defect_enhanced)
#
#         # 对图像应用中值滤波
#         median_filtered = cv2.medianBlur(defect_enhanced, 5)
#         output_path = os.path.join(output_folder, f"{image_name_prefix}_CLAFE.jpg")
#         cv2.imwrite(output_path, median_filtered)
#
#         # # 对图像应用频域低通滤波
#         # lowpass_filtered = apply_lowpass_filter(median_filtered)
#         # output_path = os.path.join(output_folder, f"{image_name_prefix}_lowpass.jpg")
#         # cv2.imwrite(output_path, lowpass_filtered)
#
#         # 可视化直方图，亮度，对比度并保存
#         visualize_hist(img, 'Original Image Histogram', output_folder, image_name_prefix)
#         visualize_hist(defect_enhanced, 'Histogram after Defect Enhancement', output_folder, image_name_prefix)
#         visualize_hist(median_filtered, 'Histogram after CLAFE', output_folder, image_name_prefix)

        # visualize_hist(lowpass_filtered, 'Histogram after Lowpass Filtering', output_folder, image_name_prefix)

# image_path = "/Users/wuzhongze/Documents/中南大学科研/2023论文发表/数据集/solar_cell_EL_image_coco/train2017"
# output_folder = "/Users/wuzhongze/Documents/中南大学科研/2023论文发表/数据集/solar_cell_EL_image_coco_data_process_last3/train2017"
# print("1")
# preprocess_images(image_path, output_folder)
#
# image_path = "/Users/wuzhongze/Documents/中南大学科研/2023论文发表/数据集/solar_cell_EL_image_coco/val2017"
# output_folder = "/Users/wuzhongze/Documents/中南大学科研/2023论文发表/数据集/solar_cell_EL_image_coco_data_process_last3/val2017"
# print("2")
# preprocess_images(image_path, output_folder)#在这个执行

# img_path = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar-data-json-coco//images//train2017//Annealed_black_spot_new_0001.jpg"
# output_folder = "/Users/wuzhongze/Documents/中南大学科研/2023论文发表/数据集/solar_cell_EL_image_coco_data_process_last4/train2017"

fig1 = "/Users/wuzhongze/Documents/中南大学科研/2023论文发表/2023论文/2023TII/实验结果记录与对比/直方图可视化/orin"
out1 = "/Users/wuzhongze/Documents/中南大学科研/2023论文发表/2023论文/2023TII/实验结果记录与对比/直方图可视化/vis"
vis_preprocess_images(fig1, out1)

#image = cv2.imread(image_path)

#threshold_h = 1000
# threshold_v = 1000

# cropped_image = crop_solar_cell_image(image, threshold_h, threshold_v)
# cv2.imwrite("path/to/your/cropped_image.jpg", cropped_image)