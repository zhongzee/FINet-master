import os
import glob

# 指定文件夹路径
folder_path = "/Users/wuzhongze/Documents/中南大学科研/2023论文发表/数据集/solar_cell_EL_image_coco/train2017"

# 指定图片文件类型
image_types = ["*.jpg", "*.jpeg", "*.png", "*.gif"]

# 构建图片文件路径模式
image_pattern = os.path.join(folder_path, "**", "*.*")
image_pattern = "|".join([os.path.join(folder_path, type) for type in image_types])

# 使用glob模块查找图片文件
image_files = glob.glob(image_pattern, recursive=True)

# 计算图片数量
num_images = len(image_files)

# 输出结果
print(f"There are {num_images} images in {folder_path}")


import os
import glob

folder_path = "/Users/wuzhongze/Documents/中南大学科研/2023论文发表/数据集/solar_cell_EL_image_coco/train2017"
image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif"]

image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(folder_path, ext)))

num_images = len(image_files)
print(f"There are {num_images} images in {folder_path}")
