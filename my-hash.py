import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import hashlib
from sklearn.neighbors import KDTree
import pickle


# 加载哈希表
filename = r"E:\anaconda3\PerceptualHashAlgorithm-master\hash_table.pkl"
with open(filename, 'rb') as file:
    hash_table = pickle.load(file)

# 加载预训练的ResNet模型
resnet = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)
# 定义预处理转换
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载和预处理查询图像
query_image_path = r'E:\anaconda3\PerceptualHashAlgorithm-master\testdata\9.jpeg'
query_image = Image.open(query_image_path).convert('RGB')
query_input_tensor = preprocess(query_image)
query_input_batch = query_input_tensor.unsqueeze(0)

# 使用ResNet模型进行前向传播，得到查询图像的特征向量
resnet.eval()
with torch.no_grad():
    query_features = resnet(query_input_batch)

# 提取查询图像的特征向量
query_feature_vector = torch.flatten(query_features, start_dim=1)

# 归一化查询特征向量
normalized_query_feature_vector = torch.nn.functional.normalize(query_feature_vector, p=2, dim=1)

# 哈希化查询特征向量
query_binary_code = hashlib.md5(normalized_query_feature_vector.numpy().tobytes()).digest()

# 构建索引结构
def build_index():
    # 提取所有特征向量和图像路径
    feature_vectors = []
    image_paths = []
    for image_list in hash_table.values():
        for image_dict in image_list:
            feature_vector = image_dict['feature_vector']
            image_path = image_dict['image_path']
            feature_vectors.append(feature_vector)
            image_paths.append(image_path)

    # 构建特征矩阵
    feature_matrix = torch.cat(feature_vectors, dim=0).numpy()

    # 构建索引结构，这里使用 KD 树
    index = KDTree(feature_matrix)

    return index, image_paths

# 构建索引结构并获取图像路径
index, image_paths = build_index()

# 在候选图像集合上使用索引进行更精确的相似性匹配
candidate_images = []
if index is not None:
    query_feature_matrix = normalized_query_feature_vector.numpy().reshape(1, -1)
    _, matching_indices = index.query(query_feature_matrix, k=5)  # 获取最相似的5个图像
    candidate_images = [image_paths[i] for i in matching_indices[0]]

print("相似图像的地址：")
for image_path in candidate_images:
    print(image_path)
# 输出相似图像
for image_path in candidate_images:
    image = Image.open(image_path)
    image.show()
