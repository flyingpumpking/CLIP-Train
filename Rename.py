import os


def rename_and_number_images(folder_path, type_str):
    # 初始化编号
    number = 1

    # 遍历文件夹中的所有文件
    for filename in sorted(os.listdir(folder_path)):
        # 检查是否是图片文件
        if filename.lower().endswith(
                ('.png', '.jpg', '.jpeg', '.gif', '.bmp')) and filename != f"{type_str}_{number}.jpg":
            file_path = os.path.join(folder_path, filename)
            # 生成新的文件名
            new_filename = f"{type_str}_{number}.jpg"
            new_file_path = os.path.join(folder_path, new_filename)

            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f'Renamed {filename} to {new_filename}')

            # 增加编号
            number += 1


# 使用示例
folder_path = r"D:\Projects\CLIP-train\datasets\toolkit-images"  # 替换为您的图片文件夹路径
type_str = 'screwdriver'  # 您可以更改这个字符串来定义类型
rename_and_number_images(folder_path, type_str)
