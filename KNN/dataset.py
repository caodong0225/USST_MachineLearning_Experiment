"""
Author: caodong0225
Date: 2024-06-04
Dataset for KNN
"""
with open("ex7data2.txt", "r", encoding="utf-8") as f:
    data_kmeans = f.readlines()
data_kmeans = [list(map(float, x.strip().split('\t'))) for x in data_kmeans]
