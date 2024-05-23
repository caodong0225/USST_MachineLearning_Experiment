import numpy as np
import random
from dataset import dataset, label
import os
np.set_printoptions(threshold=np.inf)#不显示省略号

class Ann:
    def __init__(self):

        # 创建激活函数
        # 此处罗列了一些最常用的激活函数，如果需要其他激活函数，可以自行添加
        self.active = {
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "tanh": lambda x: np.tanh(x),
            "relu": lambda x: np.where(x > 0, x, 0),
            "leakyrelu": lambda x, leaky=0.02: np.where(x > 0, x, leaky * x),
            "elu": lambda x, elu_value=0.02: np.where(x > 0, x, elu_value * (np.exp(x) - 1)),
            "softplus": lambda x: np.log(1 + np.exp(x)),
            "none": lambda x: x
        }

        # 创建损失函数
        # 此处罗列了一些最常用的损失函数，如果需要其他损失函数，可以自行添加
        self.loss = {
            "mse": lambda inp, out: np.sum((inp - out) ** 2) / len(out),
            "mae": lambda inp, out: np.sum(np.absolute(inp - out)) / len(out),
            "cross_entropy_error": lambda inp, out: -np.sum(np.where(out == 1, np.log(inp), np.log(1 - inp))) / len(out)
        }

        # 创建激活函数的导数
        self.derivative = {
            "sigmoid": lambda x: 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x))),
            "tanh": lambda x: 1 - np.tanh(x),
            "relu": lambda x: np.where(x > 0, 1, 0),
            "leakyrelu": lambda x, leaky=0.02: np.where(x > 0, 1, leaky),
            "elu": lambda x, elu_value=0.02: np.where(x > 0, 1, elu_value * np.exp(x)),
            "softplus": lambda x: 1 - 1 / (1 + np.exp(x)),
            "none": lambda x: 1
        }

        # 创建损失函数的导数
        self.derivative_loss = {
            "mse": lambda inp, out: 2 * (inp - out) / len(out),
            "mae": lambda inp, out: np.where(inp > out, 1, -1) / len(out),
            "cross_entropy_error": lambda inp, out: -np.sum(
                out * np.log(inp + 1e-7) + (1 - out) * np.log(1 - inp + 1e-7)) / len(out)
        }

        # 初始化数据
        self.grad = []
        self.bias = []

    def dot(self, inp, w, b=0, active_mod="none"):
        # 该函数用于计算输入数据与权重的点积，并且加上偏置，最后通过激活函数
        data = []
        for _ in w:
            data.append(np.sum(inp * _) + b)
        data = np.array(data)
        data = self.active[active_mod](data)
        return np.array(data)

    def initial(self, structure):
        # 初始化权重和偏置，默认为0~1之间的随机数
        self.grad = []
        self.bias = []
        for _ in range(len(structure) - 1):
            self.grad.append(np.random.randn(structure[_ + 1], structure[_]))
            self.bias.append(random.random())

    def forward(self, input_data, activate_mod):
        # 计算前向传播的结果
        layer = [np.array(input_data)]
        for _ in range(len(self.grad)):
            layer.append(self.dot(layer[_], self.grad[_], self.bias[_], activate_mod[_]))
        return layer[-1]

    def calculate_accuracy(self, input_data, expect_data, activate_mod):
        # 计算准确度
        accuracy = 0
        for ind in range(len(input_data)):
            data = input_data[ind]
            predict_data = self.forward(data, activate_mod)
            if np.argmax(predict_data) == np.argmax(expect_data[ind]):
                accuracy += 1
        accuracy = accuracy / len(input_data)
        return accuracy

    def calculate_loss(self, input_data, expect_data, activate_mod, loss_mod):
        # 计算损失
        loss = 0
        for ind in range(len(input_data)):
            data = input_data[ind]
            predict_data = self.forward(data, activate_mod)
            loss += self.loss[loss_mod](predict_data, expect_data[ind])
        loss = loss / len(input_data)
        return loss

    def renew(self, wg, bg, size, learning_rate, mod="SGD"):
        # 更新权重和偏置，通过梯度下降法
        if mod == "SGD":
            for _ in range(len(wg)):
                self.grad[_] = self.grad[_] - learning_rate * wg[_] / size
                self.bias[_] = self.bias[_] - learning_rate * bg[_] / size

    def train(self, inp, expe, stru, activemod, epoch, batchsize, lossmod, learnrate, gradmod):
        sample_size = len(expe)  # 获取样本数量
        wgrad = []
        bgrad = []
        for i in range(epoch):  # 迭代次数
            # 随机打乱数据
            index = np.random.choice(np.array(range(sample_size)), sample_size, replace=False)  # 随机打乱数据，获取打乱后的索引
            for k in range(sample_size):
                layer = []
                layer.append(np.array(inp[index[k]]))
                for j in range(len(stru) - 1):
                    layer.append(self.dot(layer[j], self.grad[j], self.bias[j], activemod[j]))

                layerback = []
                layerback.append(
                    self.derivative[activemod[-1]](self.derivative_loss[lossmod](layer[-1], expe[index[k]])) *
                    self.derivative_loss[
                        lossmod](layer[-1], expe[index[k]]))
                for j in range(len(activemod) - 1):
                    layerback.insert(0, self.derivative[activemod[-2 - j]](
                        self.dot(layerback[-1 - j], np.array(self.grad[-1 - j]).T)) * self.dot(layerback[-1 - j],
                                                                                     np.array(self.grad[-1 - j]).T))
                if len(wgrad) == 0:
                    for arr in range(len(layerback)):
                        wgrad.append(np.outer(layerback[arr], layer[arr]))
                    for arr in range(len(layerback)):
                        bgrad.append(np.sum(layerback[arr]))
                else:
                    for arr in range(len(wgrad)):
                        wgrad[arr] = wgrad[arr] + np.outer(layerback[arr], layer[arr])
                    for arr in range(len(bgrad)):
                        bgrad[arr] = bgrad[arr] + np.sum(layerback[arr])

                if (k + 1) % batchsize == 0:
                    self.renew(wgrad, bgrad, batchsize, learnrate, gradmod)
                    wgrad = []
                    bgrad = []
                elif k + 1 == len(index):
                    self.renew(wgrad, bgrad, len(index) % batchsize, learnrate, gradmod)
                    wgrad = []
                    bgrad = []

            yield 1


if __name__ == '__main__':
    model = Ann()
    model_structure = [400, 25, 10]  # 输入层400个神经元，隐藏层25个神经元，输出层10个神经元
    active_mod = ["leakyrelu", "sigmoid"]  # 第一层使用leakyrelu激活函数，第二层使用sigmoid激活函数
    loss_mod = "mse"  # 使用平方差损失函数
    # 判断是否有模型参数，如果有则加载模型参数，如果没有则初始化模型参数
    if "grad.txt" in os.listdir() and "bias.txt" in os.listdir():
        # 加载模型参数
        with open("grad.txt", "r") as f:
            model.grad = eval(f.read())
        with open("bias.txt", "r") as f:
            model.bias = eval(f.read())
    else:
        model.initial(model_structure)  # 初始化模型
    # 取数据集前4500个作为训练集
    data_train = dataset[:4500]
    label_train = label[:4500]
    # 取数据集后500个作为测试集
    data_test = dataset[500:]
    label_test = label[500:]
    gen = model.train(inp=data_train,
                      expe=label_train,
                      stru=model_structure,
                      activemod=active_mod,
                      epoch=50,
                      batchsize=128,
                      lossmod=loss_mod,
                      learnrate=0.00001,
                      gradmod="SGD")
    curr_epoch = 300
    for i in gen:
        curr_epoch += 1
        train_loss = model.calculate_loss(data_train, label_train, active_mod, loss_mod)
        train_accuracy = model.calculate_accuracy(data_train, label_train, active_mod)
        test_loss = model.calculate_loss(data_test, label_test, active_mod, loss_mod)
        test_accuracy = model.calculate_accuracy(data_test, label_test, active_mod)
        print("epoch: ", curr_epoch, end=" ")
        print("train_loss: ", train_loss, end=" ")
        print("train_accuracy: ", train_accuracy, end=" ")
        print("test_loss: ", test_loss, end=" ")
        print("test_accuracy: ", test_accuracy)
        pass
    grad_save = [list(i) for i in model.grad]
    # 保存模型参数
    with open("grad.txt", "w") as f:
        data_save = str(model.grad)
        data_save = data_save.replace("array", "np.array")
        f.write(data_save)
    with open("bias.txt", "w") as f:
        f.write(str(model.bias))
