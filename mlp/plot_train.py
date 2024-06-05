"""
绘制训练日志的图像
"""
import re
import matplotlib.pyplot as plt

with open("training.log", encoding="utf-8") as f:
    log = f.readlines()
log = [i.strip() for i in log]
# 正则表达式提取，
epoch = [int(re.findall(r"epoch:  (\d+)", i)[0]) for i in log]
train_loss = [float(re.findall(r"train_loss:  ([\d.]+)", i)[0]) for i in log]
train_accuracy = [float(re.findall(r"train_accuracy:  ([\d.]+)", i)[0]) for i in log]
test_loss = [float(re.findall(r"test_loss:  ([\d.]+)", i)[0]) for i in log]
test_accuracy = [float(re.findall(r"test_accuracy:  ([\d.]+)", i)[0]) for i in log]
# 绘图
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(epoch, train_loss, label="train_loss")
plt.plot(epoch, test_loss, label="test_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.subplot(122)
plt.plot(epoch, train_accuracy, label="train_accuracy")
plt.plot(epoch, test_accuracy, label="test_accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()
