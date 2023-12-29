# coding:utf-8
# 运行前请先做以下工作：
# pip install lxml
# 将所有的图片及xml文件存放到xml_dir指定的文件夹下，并将此文件夹放置到当前目录下


import os
import glob
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import cv2

START_BOUNDING_BOX_ID = 1
save_path = "."


def get(root, name):
    return root.findall(name)


def get_and_check(root, name, length):
    vars = get(root, name)  # 找到该root节点下所有名为name的节点列表
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(xml_list, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    categories = pre_define_categories.copy()  # 预定义
    bnd_id = START_BOUNDING_BOX_ID
    ##??
    all_categories = {}

    # 迭代处理每一个xml文件
    for index, line in enumerate(xml_list):  # 迭代处理每一个xml文件
        print("Processing %s" % (line))
        xml_f = line  # xml文件路径
        tree = ET.parse(xml_f)
        root = tree.getroot()  # 得到根节点

        # 导入每一张图片的信息
        filename = os.path.basename(xml_f)[:-4] + ".jpg"  # os.path.basename返回path最后的文件名
        image_id = 20190000001 + index
        size = get_and_check(root, 'size', 1) ##找root节点下名为“size”的节点
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}
        json_dict['images'].append(image)


        # ??Currently we do not support segmentation
        # segmented = get_and_check(root, 'segmented', 1).text
        # assert segmented == '0'

        # 导入每张图片中所有目标的信息
        for obj in get(root, 'object'):  # 迭代同一张图片中的每一个目标
            category = get_and_check(obj, 'name', 1).text
            if category in all_categories:
                all_categories[category] += 1
            else:
                all_categories[category] = 1

            # 如果出现没有预定义的目标类别
            # if category not in categories:
            # if only_care_pre_define_categories:
            # continue
            # new_id = len(categories) + 1
            # print(
            # "[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(
            #     category, pre_define_categories, new_id))
            # categories[category] = new_id

            category_id = categories[category]  # 在预定义类别中找到类别编号
            bndbox = get_and_check(obj, 'bndbox', 1)  # 找到bb节点
            # bb的左上角、右下角坐标
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            # assert (xmax > xmin), "xmax <= xmin, {}".format(line)
            # assert (ymax > ymin), "ymax <= ymin, {}".format(line)
            o_width = abs(xmax - xmin)  # bb的宽
            o_height = abs(ymax - ymin)  # bb的高

            # annotations部分
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    # categories部分
    for cate, cid in categories.items():  # dict.items()返回可遍历的(键, 值) 元组数组
        cat = {'supercategory': [], 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    json_fp = open(json_file, 'w')  # 打开json文件，写入模式
    json_str = json.dumps(json_dict, indent=4)  # json.dumps将一个Python数据结构转换为JSON字符串（将python对象编码成Json字符串）
    json_fp.write(json_str)  # 在json文件中写入建立好的Json字符串
    json_fp.close()  # 关闭json文件
    print("------------create {} done--------------".format(json_file))  # 打印json文件
    # 打印目标类别信息
    print("find {} categories: {} -->>> your pre_define_categories {}: {}".format(len(all_categories),
                                                                                  all_categories.keys(),
                                                                                  len(pre_define_categories),
                                                                                  pre_define_categories.keys()))
    print("category: id --> {}".format(categories))  # 打印目标类别id
    print(categories.keys())  # 打印目标类别
    print(categories.values())  # 类别id


if __name__ == '__main__':
    # 定义你自己的类别，5类
    # class_name = ['holothurian', 'echinus', 'starfish', 'scallop', 'waterweeds']
    # pre_define_categories = {}
    # for i, cls in enumerate(class_name):
    #     pre_define_categories[cls] = i + 1
    # 这里也可以自定义类别id，把上面的注释掉换成下面这行即可
    pre_define_categories = {"crack": 0, #改成自己的类别
                "finger": 1,
                "black_core": 2,
                "thick_line": 3,
                "star_crack": 4,
                "corner": 5,
                "fragment": 6,
                "scratch": 7,
                "horizontal_dislocation": 8,
                "vertical_dislocation": 9,
                "printing_error": 10,
                "short_circuit": 11}
    only_care_pre_define_categories = True  # 只关注预定义的类别，不关注是否有未知类别  # or False
    saved_coco_path = '/hy-tmp/mmdetection/mmdet/data/solar_cell_EL_image_coco'
    # 创建文件
    if not os.path.exists("%s/annotations/" % saved_coco_path):
        os.makedirs("%s/annotations/" % saved_coco_path)
    if not os.path.exists("%s/images/train2017/" % saved_coco_path):
        os.makedirs("%s/images/train2017" % saved_coco_path)
    if not os.path.exists("%s/images/val2017/" % saved_coco_path):
        os.makedirs("%s/images/val2017" % saved_coco_path)

    # # 保存的json文件 #改
    save_json_train = saved_coco_path + '/annotations/instances_train2017.json'
    save_json_val = saved_coco_path + '/annotations/instances_val2017.json'
    # 初始文件所在的路径 #改
    xml_train_dir = "/hy-tmp/solar_cell_EL_image/PVELAD/EL2021/trainval/Annotations"
    xml_train_list = glob.glob(xml_train_dir + "/*.xml")  # 返回所有匹配的文件路径列表
    # xml_train_list = np.sort(xml_train_list)  # 文件路径列表排序
    xml_train_jpg = '/hy-tmp/solar_cell_EL_image/PVELAD/EL2021/trainval/JPEGImages/'

    print('json_list_path: ', len(xml_train_list))
    # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
    from sklearn.model_selection import train_test_split
    train_path, val_path = train_test_split(xml_train_list, test_size=0.1, train_size=0.9)

    # 这里加上想要增加的标注的类别数据即可
    print("-" * 50)  # 打印分割线
    print("train number:", len(train_path))  # 打印训练样本数目

    print("-" * 50)  # 打印分割线
    print("val number:", len(val_path))  # 打印训练样本数目
    # # 打乱数据集
    # np.random.seed(100)  # 设置相同的seed，每次拿出的随机数就会相同；如果不设置seed，则每次会生成不同的随机数。
    # np.random.shuffle(train_path)  # shuffle() 方法将序列的所有元素随机排序，该函数没有返回值，会直接在原容器中乱序。
    print("转换训练集")
    print(save_json_train)
    convert(train_path, save_json_train)  # 参数为一个路径列表和一个保存json文件的路径
    # xml_val_dir = "/hy-tmp/solar_cell_EL_image/PVELAD/EL2021/test/Annotations"
    # xml_val_list = glob.glob(xml_val_dir + "/*.xml")  # 返回所有匹配的文件路径列表
    # xml_val_list = np.sort(val_path)  # 文件路径列表排序

    # # 打乱数据集
    # np.random.seed(100)  # 设置相同的seed，每次拿出的随机数就会相同；如果不设置seed，则每次会生成不同的随机数。
    # np.random.shuffle(val_path)  # shuffle() 方法将序列的所有元素随机排序，该函数没有返回值，会直接在原容器中乱序。
    print("转换验证集")
    print(save_json_train)
    convert(val_path, save_json_val)  # 参数为一个路径列表和一个保存json文件的路径

    print("移动训练数据集")
    for file in train_path:
        files = xml_train_jpg + file.split('/')[-1]
        # shutil.copy(file.replace("json", "jpg"), "%scoco/images/train2017/" % saved_coco_path)
        img_name = files.replace('.xml', '.jpg')
        temp_img = cv2.imread(img_name)
        try:
            cv2.imwrite("{}/images/train2017/{}".format(saved_coco_path, img_name.split('/')[-1].replace('png', 'jpg')), temp_img)
        except Exception as e:
            print(e)
            print('Wrong Image:', img_name )
            continue
        print(img_name + '-->', img_name.replace('png', 'jpg'))
    print("移动测试数据集")
    # 将验证集转换未COCO的格式
    for file in val_path:
        files = xml_train_jpg + file.split('/')[-1]
        # shutil.copy(file.replace("json", "jpg"), "%scoco/images/val2017/" % saved_coco_path)
        img_name = files.replace('.xml', '.jpg')
        temp_img = cv2.imread(img_name)
        try:
            cv2.imwrite("{}/images/val2017/{}".format(saved_coco_path, img_name.split('/')[-1].replace('png', 'jpg')), temp_img)
        except Exception as e:
            print(e)
            print('Wrong Image:', img_name)
            continue
        print(img_name + '-->', img_name.replace('png', 'jpg'))
    print("移动数据集结束")
    # import os
    # import shutil
    # print("移动")
    # source_train_path = "/hy-tmp/solar_cell_EL_image/PVELAD/EL2021/trainval/JPEGImages"
    # destination_train_path = "/hy-tmp/mmdetection/mmdet/data/solar_cell_EL_image_coco/train2017"
    #
    # source_val_path = "/hy-tmp/solar_cell_EL_image/PVELAD/EL2021/test/JPEGImages"
    # destination_val_path = "/hy-tmp/mmdetection/mmdet/data/solar_cell_EL_image_coco/val2017"
    #
    # # 获取source_path下的所有jpg文件
    # jpg_train_files = [f for f in os.listdir(source_train_path) if f.endswith('.jpg')]
    #
    # # 将所有jpg文件复制到destination_path
    # for jpg_file in jpg_train_files:
    #     source_file = os.path.join(source_train_path, jpg_file)
    #     dest_file = os.path.join(destination_train_path, jpg_file)
    #     shutil.copyfile(source_file, dest_file)
    #
    # # 获取source_path下的所有jpg文件
    # jpg_val_files = [f for f in os.listdir(source_val_path) if f.endswith('.jpg')]
    #
    # # 将所有jpg文件复制到destination_path
    # for jpg_file in jpg_val_files:
    #     source_file = os.path.join(source_val_path, jpg_file)
    #     dest_file = os.path.join(destination_val_path, jpg_file)
    #     shutil.copyfile(source_file, dest_file)
    # print("移动结束")

