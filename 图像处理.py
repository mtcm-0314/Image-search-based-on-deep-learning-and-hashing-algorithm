import cv2
import os

def normalize_image_size(image_library, width, height):
    for image_path in image_library:
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (width, height))
        cv2.imwrite(image_path, resized_image)

# 图像库路径
image_library_path = r"D:\Pytouch\PerceptualHashAlgorithm-master\target_bag"

# 获取图像库中的所有图像文件路径
image_library = [os.path.join(image_library_path, file) for file in os.listdir(image_library_path) if file.endswith((".jpg", ".png"))]

# 指定归一化后的图像大小
normalized_width = 256
normalized_height = 256

# 调用函数进行图像归一化
normalize_image_size(image_library, normalized_width, normalized_height)