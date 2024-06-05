"""
Author: caodong0225
Date: 2024-06-05
Description: 读取数据集部分

"""
import numpy as np
with open("data.txt", encoding="utf-8") as f:
    data_read = f.readlines()
dataset = []
for _ in data_read:
    dataset.append(list(map(float, _.split())))
with open("label.txt", encoding="utf-8") as f:
    data_read = f.readlines()
label = []
for _ in data_read:
    data_temp = [0]*10
    data_temp[int(_)-1] = 1
    label.append(data_temp)
dataset = np.array(dataset)
label = np.array(label)
# 打乱数据
temp = list(zip(dataset, label))
np.random.shuffle(temp)
dataset, label = zip(*temp)
dataset = np.array(dataset)
label = np.array(label)
