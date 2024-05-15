# 读取数据集
with open("ex2data2.txt") as f:
    data = f.readlines()
data = [line.strip().split(",") for line in data]
for _ in data:
    for i in range(len(_)):
        _[i] = float(_[i])
