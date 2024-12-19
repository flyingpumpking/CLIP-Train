import os


def get_image_paths(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths


# 指定路径
img_folder = r"D:\Projects\CLIP-train\datasets\part-images"
img_paths = get_image_paths(img_folder)

for path in img_paths:
    print(path)
