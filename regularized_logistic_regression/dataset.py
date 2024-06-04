"""
训练集
"""
# 读取数据集
with open("ex2data2.txt", encoding="utf-8") as f:
    data = f.readlines()
data = [line.strip().split(",") for line in data]
for _ in data:
    for index, value in enumerate(_):
        _[index] = float(value)
