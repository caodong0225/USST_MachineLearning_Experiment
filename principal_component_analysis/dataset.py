"""
Author: caodong0225
Date: 2024-06-04
Description: 数据集加载
"""
import numpy as np
with open("ex7data1.txt", "r", encoding="utf-8") as f:
    data_pca = f.readlines()
data_pca = [list(map(float, x.strip().split('\t'))) for x in data_pca]
with open("ex7faces.txt", "r", encoding="utf-8") as f:
    data_faces = f.readlines()
data_faces = [list(map(float, x.strip().split('\t'))) for x in data_faces]
data_faces = np.array(data_faces)
# print(data_faces.shape)  # (5000, 1024)
data_faces = data_faces.reshape((5000, 32, 32))
# 将data_faces旋转90度
data_faces = np.rot90(data_faces, k=3, axes=(1, 2))
