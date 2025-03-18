import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import hashlib
from sklearn.neighbors import KDTree
import pickle

# 创建哈希表
hash_table = {}

# 加载预训练的ResNet模型
resnet = models.resnet50(weights=models.resnet.ResNet50_Weights.IMAGENET1K_V1)

# 定义预处理转换
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载和预处理图像，并构建哈希表和索引结构
def add_image_to_index(image_path):
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # 使用ResNet模型进行前向传播
    resnet.eval()
    with torch.no_grad():
        features = resnet(input_batch)

    # 提取特征向量
    feature_vector = torch.flatten(features, start_dim=1)

    # 归一化特征向量
    normalized_feature_vector = torch.nn.functional.normalize(feature_vector, p=2, dim=1)

    # 哈希化特征向量
    binary_code = hashlib.md5(normalized_feature_vector.numpy().tobytes()).digest()

    # 将特征向量存储在哈希表中
    if binary_code in hash_table:
        hash_table[binary_code].append({
            'image_path': image_path,
            'feature_vector': normalized_feature_vector
        })
    else:
        hash_table[binary_code] = [{
            'image_path': image_path,
            'feature_vector': normalized_feature_vector
        }]

# 图像文件夹路径
image_folder = r'E:\anaconda3\PerceptualHashAlgorithm-master\testdata'

# 遍历图像文件夹，并将图像添加到哈希表中
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    add_image_to_index(image_path)

# 储存哈希表到文件
filename = 'hash_table.pkl'
with open(filename, 'wb') as file:
    pickle.dump(hash_table, file)

# 构建特征矩阵和索引结构
def build_index():
    # 提取所有特征向量
    feature_vectors = []
    for image_list in hash_table.values():
        for image_dict in image_list:
            feature_vector = image_dict['feature_vector']
            feature_vectors.append(feature_vector)

    # 构建特征矩阵
    feature_matrix = torch.cat(feature_vectors, dim=0).numpy()

    # 构建索引结构，这里使用 KD 树
    index = KDTree(feature_matrix)

    return index

# 加载和预处理查询图像
query_image_path = r'E:\anaconda3\PerceptualHashAlgorithm-master\train_data\cat\913.jpeg'
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

# 在哈希表中查找与查询哈希码相似的候选图像
candidate_images = hash_table.get(query_binary_code, [])

# 构建索引结构
index = build_index()

# 在候选图像集合上使用索引进行更精确的相似性匹配
matching_images = []
if index is not None and candidate_images:
    query_feature_matrix = normalized_query_feature_vector.numpy().reshape(1, -1)
    _, matching_indices = index.query(query_feature_matrix, k=1)
    matching_images = [candidate_images[i] for i in matching_indices[0]]

#这是一个完整的示例代码，它包含了创建哈希表、加载模型、预处理图像、计算特征向量、哈希化特征向量、储存哈希表到文件、构建特征矩阵和索引结构、加载和预处理查询图像、以及在哈希表和索引结构中进行相似性匹配的逻辑。

#请注意，你需要将以下路径更改为适合你的实际数据的路径：
#- `image_folder`：图像文件夹的路径
#- `query_image_path`：查询图像的路径

#此外，确保你已经安装了所需的依赖项，如`torch`、`torchvision`、`opencv-python`和`scikit-learn`。