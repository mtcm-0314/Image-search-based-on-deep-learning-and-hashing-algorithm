"""
以图搜图：感知哈希算法（Perceptual Hash Algorithm，简称pHash）的原理与实现
测试环境：win10 | python 3.9.13 | OpenCV 4.4.0 | numpy 1.21.1
实验时间：2023-10-22
"""

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def get_hash(img_path):
    # 读取图像：通过OpenCV的imread加载图像
    # 缩小图像：使用OpenCV的resize函数将图像缩放为32x32像素，采用Cubic插值方法进行图像重采样
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32), cv2.INTER_CUBIC)

    # 图像灰度化：将彩色图像转换为灰度图像。
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(f"缩放32x32的图像中每个像素的颜色=\n{img_gray}")

    # 离散余弦变换（DCT）：计算图像的DCT变换，得到32×32的DCT变换系数矩阵
    img_dct = cv2.dct(np.float32(img_gray))
    # print(f"灰度图像离散余弦变换（DCT）={img_dct}")

    # 缩放DCT：将DCT系数的大小显式地调整为8x8。然后它计算调整后的DCT系数的均值，并生成哈希值。
    # dct_roi = img_dct[0:8, 0:8]
    img_dct.resize(8, 8)

    # 计算灰度均值：计算DCT变换后图像块的均值
    img_avg = np.mean(img_dct)
    # print(f"DCT变换后图像块的均值={img_avg}")

    """
    # # 生成二进制哈希值
    # img_hash_str = ''
    # for i in range(8):
    #     for j in range(8):
    #         if img_dct[i, j] > img_avg:
    #             img_hash_str += '1'
    #         else:
    #             img_hash_str += '0'
    # print(f"图像的二进制哈希值={img_hash_str}")

    # # 生成图像可识别哈希值
    # img_hash = ''
    # for i in range(0, 64, 4):
    #     img_hash += ''.join('%x' % int(img_hash_str[i: i + 4], 2))
    # print(f"图像可识别的哈希值={img_hash}")
    """

    img_hash_str = ""
    for i in range(8):
        img_hash_str += ''.join(map(lambda i: '0' if i < img_avg else '1', img_dct[i]))
    # print(f"图像的二进制哈希值={img_hash_str}")

    # 生成图像可识别哈希值
    img_hash = ''.join(map(lambda x: '%x' % int(img_hash_str[x: x + 4], 2), range(0, 64, 4)))
    # print(f"图像可识别的哈希值={img_hash}")
    return img_hash


# 汉明距离：计算两个等长字符串（通常是二进制字符串或位字符串）之间的汉明距离。用于确定两个等长字符串在相同位置上不同字符的数量。
def hamming_distance(s1, s2):
    # 检查这两个字符串的长度是否相同。如果长度不同，它会引发 ValueError 异常，因为汉明距离只适用于等长的字符串
    if len(s1) != len(s2):
        raise ValueError("Input strings must have the same length")

    distance = 0
    for i in range(len(s1)):
        # 遍历两个字符串的每个字符，比较它们在相同位置上的值。如果发现不同的字符，将 distance 的值增加 1
        if s1[i] != s2[i]:
            distance += 1
    return distance


# --------------------------------------------------------- 测试 ---------------------------------------------------------

time_start = time.time()

img_1 = 'img_test/apple-01.jpg'
img_2 = 'img_test/apple-02.jpg'
img_3 = 'img_test/apple-03.jpg'
img_4 = 'img_test/apple-04.jpg'
img_5 = 'img_test/apple-05.jpg'
img_6 = 'img_test/apple-06.jpg'
img_7 = 'img_test/apple-07.jpg'
img_8 = 'img_test/apple-08.jpg'
img_9 = 'img_test/apple-09.jpg'
img_10 = 'img_test/pear-001.jpg'

img_hash1 = get_hash(img_1)
img_hash2 = get_hash(img_2)
img_hash3 = get_hash(img_3)
img_hash4 = get_hash(img_4)
img_hash5 = get_hash(img_5)
img_hash6 = get_hash(img_6)
img_hash7 = get_hash(img_7)
img_hash8 = get_hash(img_8)
img_hash9 = get_hash(img_9)
img_hash10 = get_hash(img_10)

distance1 = hamming_distance(img_hash1, img_hash1)
distance2 = hamming_distance(img_hash1, img_hash2)
distance3 = hamming_distance(img_hash1, img_hash3)
distance4 = hamming_distance(img_hash1, img_hash4)
distance5 = hamming_distance(img_hash1, img_hash5)
distance6 = hamming_distance(img_hash1, img_hash6)
distance7 = hamming_distance(img_hash1, img_hash7)
distance8 = hamming_distance(img_hash1, img_hash8)
distance9 = hamming_distance(img_hash1, img_hash9)
distance10 = hamming_distance(img_hash1, img_hash10)

time_end = time.time()

print(f"图片名称：{img_1}，图片HASH：{img_hash1}，与图片1的近似值（汉明距离）：{distance1}")
print(f"图片名称：{img_2}，图片HASH：{img_hash2}，与图片1的近似值（汉明距离）：{distance2}")
print(f"图片名称：{img_3}，图片HASH：{img_hash3}，与图片1的近似值（汉明距离）：{distance3}")
print(f"图片名称：{img_4}，图片HASH：{img_hash4}，与图片1的近似值（汉明距离）：{distance4}")
print(f"图片名称：{img_5}，图片HASH：{img_hash5}，与图片1的近似值（汉明距离）：{distance5}")
print(f"图片名称：{img_6}，图片HASH：{img_hash6}，与图片1的近似值（汉明距离）：{distance6}")
print(f"图片名称：{img_7}，图片HASH：{img_hash7}，与图片1的近似值（汉明距离）：{distance7}")
print(f"图片名称：{img_8}，图片HASH：{img_hash8}，与图片1的近似值（汉明距离）：{distance8}")
print(f"图片名称：{img_9}，图片HASH：{img_hash9}，与图片1的近似值（汉明距离）：{distance9}")
print(f"图片名称：{img_10}，图片HASH：{img_hash10}，与图片1的近似值（汉明距离）：{distance10}")

print(f"耗时：{time_end - time_start}")
