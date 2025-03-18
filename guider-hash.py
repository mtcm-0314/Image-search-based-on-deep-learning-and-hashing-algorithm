# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:31:57 2019

@author: wuzhe
"""
import os

import cv2
from math import floor
import numpy as np
import dhash
from PIL import Image
from matplotlib import pyplot as plt

"""
author: zhenyu wu
time: 2019/12/04 16:03
function: 均值哈希距离计算函数
params: 
    img: 输入的图片
return:
    temp: 均值哈希指纹计算结果
"""


def HashValue(img):
    # 均值哈希指纹计算函数

    # 调整图片大小为8x8像素，使用双立方插值法进行插值
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)

    # 将图片转换为灰度图像
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 对每个像素的灰度值除以4，取整数部分
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = img[i, j] // 4

    # 计算灰度值的平均值
    avg = np.sum(img) / 64 * np.ones((8, 8))

    # 计算差值，大于等于0的像素置为1，小于0的像素置为0
    temp = img - avg
    temp[temp >= 0] = 1
    temp[temp < 0] = 0

    # 将二维数组转换为一维数组
    temp = temp.reshape((1, 64))

    # 返回均值哈希指纹计算结果
    return temp


"""
author: zhenyu wu
time: 2019/12/04 16:04
function: 根据均值哈希算法计算的汉明距离
params: 
    img1: 输入的图片
    img2: 输入的图片
return:
    result: 汉明距离计算结果
"""


def Hash(img1, img2):
    img1 = HashValue(img1)
    img2 = HashValue(img2)
    result = np.nonzero(img1 - img2)
    result = np.shape(result[0])[0]
    if result <= 5:
        print('Same Picture')
    return result


"""
author: zhenyu wu
time: 2019/12/04 16:06
function: 感知哈希距离计算函数
params: 
    img: 输入的图片
return:
    temp: 感知哈希指纹计算结果
"""


def pHashValue(img):
    # 感知哈希指纹计算函数

    # 调整图片大小为32x32像素，使用双立方插值法进行插值
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    #使用不同的插值法会得到不同的结果

    # 将图片转换为灰度图像
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 将图像数据类型转换为float32
    img = img.astype(np.float32)

    # 进行离散余弦变换（DCT）
    img = cv2.dct(img)

    # 取DCT系数的左上角8x8区域
    img = img[:8, :8]
    # plt.imshow(img, cmap='gray')
    # plt.show()

    # 计算DCT系数的平均值
    avg = np.sum(img) / 64 * np.ones((8, 8))

    # 计算差值，大于等于0的系数置为1，小于0的系数置为0
    temp = img - avg
    temp[temp >= 0] = 1
    temp[temp < 0] = 0

    # 将二维数组转换为一维数组
    temp = temp.reshape((1, 64))

    # 返回感知哈希指纹计算结果
    return temp

"""
author: zhenyu wu
time: 2019/12/04 16:06
function: 根据感知哈希算法计算的汉明距离
params: 
    img1: 输入的图片
    img2: 输入的图片
return:
    result: 汉明距离计算结果
"""


def pHash(img1, img2):
    img1 = pHashValue(img1)
    img2 = pHashValue(img2)
    result = np.nonzero(img1 - img2)
    result = np.shape(result[0])[0]
    if result <= 5:
        print('Same Picture')
    return result


"""
author: zhenyu wu
time: 2019/12/09 09:14
function: 差值哈希距离计算函数
params: 
    img: 输入的图片
return:
    temp: 差值哈希指纹计算结果
"""


def DHashValue(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    img2 = []
    for i in range(8):
        img2.append(np.array(img[:, i]) - np.array(img[:, i + 1]))
    img2 = np.mat(img2).T
    img2[img2 >= 0] = 1
    img2[img2 < 0] = 0
    img2 = img2.reshape((1, 64))
    return img2


"""
author: zhenyu wu
time: 2019/12/09 09:13
function: 根据差值哈希算法计算的汉明距离
params: 
    img1: 输入的图片
    img2: 输入的图片
return:
    result: 汉明距离计算结果
"""


def DHash(img1, img2):
    img1 = DHashValue(img1)
    img2 = DHashValue(img2)
    result = np.nonzero(img1 - img2)
    result = np.shape(result[0])[0]
    if result <= 5:
        print('Same Picture')
    return result


"""
author: zhenyu wu
time: 2019/12/09 09:37
function: 根据包中的差值哈希算法计算的汉明距离
params: 
    img1: 输入的图片
    img2: 输入的图片
return:
    result: 汉明距离计算结果
"""


def dHash_use_package(img1, img2):
    image1 = Image.open(img1)
    image2 = Image.open(img2)
    row1, col1 = dhash.dhash_row_col(image1)
    row2, col2 = dhash.dhash_row_col(image2)
    a1 = int(dhash.format_hex(row1, col1), 16)
    a2 = int(dhash.format_hex(row2, col2), 16)
    result = dhash.get_num_bits_different(a1, a2)
    if result <= 5:
        print('Same Picture')
    return result


def build_image_library(directory):
    image_library = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image_library.append(image_path)
    return image_library

def search_similar_image(target_image_path, image_library):
    target_image = cv2.imread(target_image_path)
    target_hash = HashValue(target_image)
    phash = pHashValue(target_image)
    dhash = DHashValue(target_image)

    distances = []
    for image_path in image_library:
        image = cv2.imread(image_path)
        hash_distance = Hash(target_image, image)
        phash_distance = pHash(target_image, image)
        dhash_distance = DHash(target_image, image)
        dhash_pkg_distance = dHash_use_package(target_image_path, image_path)

        distances.append((image_path, hash_distance, phash_distance, dhash_distance, dhash_pkg_distance))

    # 根据不同的哈希距离排序
    distances.sort(key=lambda x: x[1])  # 按照均值哈希距离排序
    print("按照均值哈希距离排序：")
    most_similar_image_path, min_distance = distances[0][:2]
    most_similar_image = cv2.imread(most_similar_image_path)
    cv2.imshow("最相似的图像（均值哈希）", most_similar_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"最相似的图像（均值哈希）：{most_similar_image_path}")
    print(f"距离：{min_distance}")
    print()

    distances.sort(key=lambda x: x[2])  # 按照感知哈希距离排序
    print("按照感知哈希距离排序：")
    most_similar_image_path, min_distance = distances[0][:2]
    most_similar_image = cv2.imread(most_similar_image_path)
    cv2.imshow("最相似的图像（感知哈希）", most_similar_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"最相似的图像（感知哈希）：{most_similar_image_path}")
    print(f"距离：{min_distance}")
    print()

    distances.sort(key=lambda x: x[3])  # 按照差值哈希距离排序
    print("按照差值哈希距离排序：")
    most_similar_image_path, min_distance = distances[0][:2]
    most_similar_image = cv2.imread(most_similar_image_path)
    cv2.imshow("最相似的图像（差值哈希）", most_similar_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"最相似的图像（差值哈希）：{most_similar_image_path}")
    print(f"距离：{min_distance}")
    print()

    distances.sort(key=lambda x: x[4])  # 按照差值哈希（使用包）距离排序
    print("按照差值哈希（使用包）距离排序：")
    most_similar_image_path, min_distance = distances[0][:2]
    most_similar_image = cv2.imread(most_similar_image_path)
    cv2.imshow("最相似的图像（差值哈希使用包）", most_similar_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"最相似的图像（差值哈希使用包）：{most_similar_image_path}")
    print(f"距离：{min_distance}")
    print()





def main():
    # 图像库目录
    image_library_directory = 'image_bag'

    # 构建图像库
    image_library = build_image_library(image_library_directory)

    # 目标图像
    target_image_path = r'D:\Pytouch\PerceptualHashAlgorithm-master\target_bag\target_image2.jpg'

    # 搜索最相近的图像并输出
    search_similar_image(target_image_path, image_library)

if __name__ == '__main__':
    main()