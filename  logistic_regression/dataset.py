"""
Author: caodong0225
Date: 2024-06-04
Description: 读取数据集
"""
# 读取数据集
with open("ex2data1.txt", encoding="utf-8") as f:
    data = f.readlines()
data = [line.strip().split(",") for line in data]
for _ in data:
    for index, __ in enumerate(_):
        _[index] = float(__)
