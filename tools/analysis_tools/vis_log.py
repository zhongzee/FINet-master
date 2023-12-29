import matplotlib.pyplot as plt
import json


def visualize_json_keys(file_paths, keys, labels, x_key='iter', save_path=None):
    """
    Visualize the specified keys from JSON files.

    :param file_paths: list of file paths to the JSON files.
    :param keys: list of keys to visualize.
    :param labels: list of labels for the files.
    :param x_key: key to use as the x-axis, default is 'iter'.
    :param save_path: path to save the plot, if None, the plot is not saved.
    """

    # 不同的线型用于不同的键值
    linestyles = ['-', '--', '-.', ':']

    # 对于每个文件
    for file_index, file_path in enumerate(file_paths):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")

        # 对于每个键值
        for key_index, key in enumerate(keys):
            x_values = []
            y_values = []
            for entry in data:
                if x_key in entry and key in entry:
                    x_values.append(entry[x_key])
                    y_values.append(entry[key])

            # 绘制曲线
            plt.plot(x_values, y_values, label=f'{labels[file_index]}: {key}',
                     color=f'C{file_index}', linestyle=linestyles[key_index % len(linestyles)])

    # 添加图例和标签
    plt.legend()
    plt.xlabel(x_key)
    plt.ylabel('Values')
    plt.title('Visualization of JSON Keys')
    plt.show()

    # 保存图像
    if save_path is not None:
        plt.savefig(save_path)


# 示例使用
# visualize_json_keys(['/path/to/file1.json', '/path/to/file2.json'], ['key1', 'key2'], ['Label1', 'Label2'], save_path='/path/to/save.png')


# 示例使用
# visualize_json_keys(['/path/to/file1.json', '/path/to/file2.json'], ['key1', 'key2'], ['Label1', 'Label2'])

"""
/Users/wuzhongze/Documents/中南大学科研/2023论文发表/2023论文/2023半监督期刊/实验/consistent_teacher_r50_fpn_voc0924_72k_baseline/20230924_111016.log.json
/Users/wuzhongze/Documents/中南大学科研/2023论文发表/2023论文/2023半监督期刊/实验/consistent_teacher_r50_fpn_voc0924_72k_mha_fam3d/20230926_230506.log.json
"""
path1 = "/Users/wuzhongze/Documents/中南大学科研/2023论文发表/2023论文/2023半监督期刊/实验/consistent_teacher_r50_fpn_voc0924_72k_baseline/20230924_111016.log.json"
path2 = "/Users/wuzhongze/Documents/中南大学科研/2023论文发表/2023论文/2023半监督期刊/实验/consistent_teacher_r50_fpn_voc0924_72k_mha_fam3d/20230926_230506.log.json"
key1 = "teacher.bbox_mAP"
key2 = "student.bbox_mAP"
label1 = "fam3d"
label1 = "mha_fam3d"
# 示例使用
visualize_json_keys(['/path/to/file1.json', '/path/to/file2.json'], ['key1', 'key2'], ['Label1', 'Label2'], save_path='/path/to/save.png')


