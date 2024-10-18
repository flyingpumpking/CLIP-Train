import os
from PIL import Image


def convert_images_to_jpg_and_delete_original(folder_path):
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.jpg'):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path)
                new_filename = os.path.splitext(file_path)[0] + '.jpg'
                img.convert('RGB').save(new_filename, 'jpeg')
                os.remove(file_path)  # 删除原始文件
                print(f'Converted and deleted original: {filename}')
            except Exception as e:
                print(f'Failed to convert {filename}: {e}')


# 使用示例
folder_path = r"D:\Projects\CLIP-train\datasets\toolkit-images"  # 将此路径替换为您的图片文件夹路径
convert_images_to_jpg_and_delete_original(folder_path)
