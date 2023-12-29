import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import shutil

def rotate_image_to_horizontal(image, edge_threshold=100, line_threshold=150):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    edges = cv2.Canny(gray_image, edge_threshold, edge_threshold * 3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, line_threshold)

    max_length = 0
    max_angle = 0
    if lines is not None:
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

    rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), max_angle, 1)
    result = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return result, max_angle


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

    return cropped_image,left_boundary,right_boundary,top_boundary,bottom_boundary

def visualize_and_save(img, title, output_folder):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.savefig(os.path.join(output_folder, f"{title}.png"))
    plt.show()

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

    return cropped_image,left_boundary,right_boundary


def update_bboxes(annotations, image_id, angle, left_boundary, top_boundary):
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            bbox = annotation['bbox']
            x, y, w, h = bbox
            # 将边界框的中心点坐标转换为NumPy数组
            bbox_center = np.array([[x + w / 2], [y + h / 2], [1]])

            # 创建旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D((0, 0), -angle, 1)

            # 将边界框中心点坐标旋转
            rotated_bbox_center = np.matmul(rotation_matrix, bbox_center)
            x_rotated, y_rotated = rotated_bbox_center[0, 0], rotated_bbox_center[1, 0]

            # 裁剪边界框中心点坐标
            x_cropped = x_rotated - left_boundary
            y_cropped = y_rotated - top_boundary

            # 更新边界
            # 更新边界框信息
            annotation['bbox'] = [x_cropped, y_cropped, w, h]

            # 如果边界框的坐标值小于零，则将其设置为零
            for i in range(2):
                if annotation['bbox'][i] < 0:
                    annotation['bbox'][i] = 0


def process_images_and_update_annotations(input_image_directory, output_image_directory):
    skipped_images = []
    output_folder = "//Users/wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar_cell_image_coco_process_tmp//"
    # 如果目录不存在，就创建目录
    if not os.path.exists(output_image_directory):
        os.makedirs(output_image_directory)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_image_directory) if f.endswith('.jpg') or f.endswith('.jpeg')]

    for image_file in image_files:
        try:
            json_file = os.path.splitext(image_file)[0] + '.json'
            image_path = os.path.join(input_image_directory, image_file)
            json_path = os.path.join(input_image_directory, json_file)
            image_name_prefix = os.path.splitext(os.path.basename(image_path))[0]
            with open(json_path, 'r') as f:
                image_info = json.load(f)

            image = cv2.imread(image_path)
            # 自适应旋转
            rotated_image, max_angle = rotate_image_to_horizontal(image)
            # visualize_and_save(rotated_image, f'{image_name_prefix}_01_rotate', output_folder)
            # 动态自适应的裁剪
            cropped_image,left_boundary,right_boundary,top_boundary,bottom_boundary = dynamic_crop_solar_cell_image(rotated_image)
            # visualize_and_save(cropped_image, f'{image_name_prefix}_02_cropped_image', output_folder)
            # 自适应直方图均衡化
            if len(cropped_image.shape) == 3:
                img_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = cropped_image

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            res_clahe = clahe.apply(img_gray)

            # 中值滤波
            final_image = cv2.medianBlur(res_clahe, 5)

            output_image_path = os.path.join(output_image_directory, image_file)
            cv2.imwrite(output_image_path, final_image)

            # 更新图像尺寸和边界框信息
            image_info['imageHeight'], image_info['imageWidth'] = final_image.shape[:2]

            for shape in image_info['shapes']:
                x, y, w, h = shape['points'][0][0], shape['points'][0][1], shape['points'][1][0] - shape['points'][0][
                    0], shape['points'][1][1] - shape['points'][0][1]

                # 计算旋转后的边界框坐标
                left_top = np.array([x, y, 1]).reshape(-1, 1)
                right_bottom = np.array([x + w, y + h, 1]).reshape(-1, 1)

                # 使用负最大角度计算旋转矩阵
                rotation_matrix_2d = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), -max_angle, 1)

                # 将旋转矩阵转换为3x3矩阵
                rotation_matrix = np.vstack([rotation_matrix_2d, [0, 0, 1]])

                # 旋转边界框坐标
                left_top_rotated = rotation_matrix.dot(left_top)
                right_bottom_rotated = rotation_matrix.dot(right_bottom)

                cropped_top_left = np.array([left_boundary, top_boundary])
                # cropped_bottom_right = np.array([right_boundary, bottom_boundary])

                left_top_cropped = left_top_rotated[:2] - cropped_top_left
                right_bottom_cropped = right_bottom_rotated[:2] - cropped_top_left

                # 转换回整数坐标
                left_top_cropped = left_top_cropped.astype(int)
                right_bottom_cropped = right_bottom_cropped.astype(int)

                # 获取旋转和裁剪后的边界框坐标
                x_cropped, y_cropped = left_top_cropped[0, 0], left_top_cropped[1, 0]
                w_cropped, h_cropped = right_bottom_cropped[0, 0] - left_top_cropped[0, 0], right_bottom_cropped[1, 0] - left_top_cropped[1, 0]

                x_cropped = int(x_cropped)
                y_cropped = int(y_cropped)
                w_cropped = int(w_cropped)
                h_cropped = int(h_cropped)

                # 修改边界框坐标
                shape['points'] = [[x_cropped, y_cropped], [x_cropped + w_cropped, y_cropped + h_cropped]]

                # 保存更新后的JSON文件
                output_json_path = os.path.join(output_image_directory, json_file)
                with open(output_json_path, 'w') as f:
                    json.dump(image_info, f, indent=2)
                # print(f"Processed {image_file} successfully.")

        except Exception as e:
            print(f"Error processing image: {image_file}. Skipping...")
            print(e)
            skipped_images.append(image_file)
            continue
    print(f"Processed {len(image_files) - len(skipped_images)} images out of {len(image_files)}")

def process_images(input_image_directory_val, output_image_directory_val):
    if not os.path.exists(output_image_directory_val):
        os.makedirs(output_image_directory_val)
    out_tmp = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar_cell_EL_image_coco_data_process_tmp"
    if not os.path.exists(out_tmp):
        os.makedirs(out_tmp)
    image_name_prefix = os.path.splitext(os.path.basename(input_image_directory_val))[0]
    image_files = [f for f in os.listdir(input_image_directory_val) if f.endswith('.jpg') or f.endswith('.jpeg')]
    skipped_images = []
    for image_file in image_files:
        try:
            image_path = os.path.join(input_image_directory_val, image_file)
            image = cv2.imread(image_path)

            # 自适应直方图均衡化
            if len(image.shape) == 3:
                # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = image
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # res_clahe = clahe.apply(img)
            # # 中值滤波
            # median_filtered = cv2.medianBlur(img, 5)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            res_clahe = clahe.apply(img)
            visualize_and_save(res_clahe, f'{image_name_prefix}_01_res_clahe', out_tmp)
            # final_image = res_clahe
            # 中值滤波
            final_image = cv2.medianBlur(res_clahe, 5)
            visualize_and_save(res_clahe, f'{image_name_prefix}_02_medianBlur',out_tmp)
            output_image_path = os.path.join(output_image_directory_val, image_file)
            cv2.imwrite(output_image_path, final_image)

            # # # 复制标注文档
            # json_file = os.path.splitext(image_file)[0] + '.json'
            # input_json_path = os.path.join(input_image_directory_val, json_file)
            # output_json_path = os.path.join(output_image_directory_val, json_file)
            # shutil.copy(input_json_path, output_json_path)
        except Exception as e:
            print(f"Error processing image: {image_file}. Skipping...")
            print(e)
            skipped_images.append(image_file)
            continue
# input_image_directory_train = "path/to/your/train"
# output_image_directory_train = "path/to/output/train"
#
# input_image_directory_val = "path/to/your/val"
# output_image_directory_val = "path/to/output/val"
#
input_image_directory_train = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar_cell_EL_image_coco_process//train2017"
input_image_directory_val = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar_cell_EL_image_coco//val2017"
output_image_directory_train = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar_cell_EL_image_coco_data_process5/train2017"
output_image_directory_val = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar_cell_EL_image_coco_data_process5//val2017"

# process_images_and_update_annotations(input_image_directory_train, output_image_directory_train)
# print("处理验证集开始...")
# process_images_and_update_annotations(input_image_directory_val, output_image_directory_val)
# print("处理验证集结束...")

print("处理训练集开始...")
process_images(input_image_directory_train, output_image_directory_train)
print("处理训练集结束...")

print("处理验证集开始...")
process_images(input_image_directory_val, output_image_directory_val)
print("处理验证集结束...")

# 指定输入和输出目录
# input_image_directory_train = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar-data-json-coco//images//train2017"
# input_image_directory_val = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar-data-json"
# output_image_directory_train = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar-data-json-coco-process/train2017"
# output_image_directory_val = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar-data-json-process"
#
# # 指定输入和输出标注文件
# input_json_train = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar-data-json-coco//annotations//instances_train2017.json"
# input_json_val = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar-data-json-coco//annotations//instances_val2017.json"
# output_json_train = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar-data-json-coco-process//annotations//instances_train2017.json"
# output_json_val = "//Users//wuzhongze//Documents//中南大学科研//2023论文发表//数据集//solar-data-json-coco-process//annotations//instances_val2017.json"
#
# # 处理训练集
# # print("处理训练集开始...")
# # process_images_and_update_annotations(input_image_directory_train, input_json_train, output_image_directory_train, output_json_train)
# # print("处理验证集结束...")
#
# # # 处理验证集
# print("处理验证集开始...")
# process_images_and_update_annotations(input_image_directory_val, input_json_val, output_image_directory_val, output_json_val)
# print("处理验证集结束...")