"""
Author: caodong0225
Date: 2024-06-04
Description: 训练主成分分析
"""
import numpy as np
from dataset import data_faces
from plot_data import show_faces


def pca(x_, k_):
    """
    :param x_:
    :param k_:
    :return:
    """
    # 标准化数据：减去均值
    x_mean = np.mean(x_, axis=0)
    x_centered = x_ - x_mean

    # 计算协方差矩阵
    covariance_matrix = np.cov(x_centered, rowvar=False)

    # 使用SVD计算特征值和特征向量
    u, _, __ = np.linalg.svd(covariance_matrix)

    # 选择前k个主成分
    u_reduced = u[:, :k_]

    # 将数据投影到前k个主成分上
    x_reduced = np.dot(x_centered, u_reduced)

    return x_reduced, u_reduced, x_mean


def recover_data(x_reduced, u_reduced, x_mean):
    """
    :param x_reduced:
    :param u_reduced:
    :param x_mean:
    :return:
    """
    return np.dot(x_reduced, u_reduced.T) + x_mean


# 示例数据
X = data_faces.reshape((5000, 1024))

# 将特征压缩到100维度
k = 100
X_reduced, U_reduced, X_mean = pca(X, k)
# print(X_reduced.shape)  # (5000, 100)
X_recovers = recover_data(X_reduced, U_reduced, X_mean)
# print(X_recovers.shape)  # (5000, 1024)
X_recovers = X_recovers.reshape((5000, 32, 32))
show_faces(X_recovers)
X_reduced = X_reduced.reshape((5000, 10, 10))
show_faces(X_reduced)
