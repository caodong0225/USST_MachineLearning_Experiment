"""
Author: caodong0225
Date: 2024-06-05
Description: 训练主成分分析
"""
import numpy as np
import matplotlib.pyplot as plt
from dataset import data_pca


# Function to normalize the dataset
def feature_normalize(x_):
    """
    :param x_:
    :return:
    """
    mu_ = np.mean(x_, axis=0)
    sigma_ = np.std(x_, axis=0)
    x_norm_ = (x_ - mu_) / sigma_
    return x_norm_, mu_, sigma_


# PCA主成分分析
def pca(x_, k):
    """
    :param x_:
    :param k:
    :return:
    """
    m, _ = x_.shape
    sigma_ = x_.T.dot(x_) / m
    u_, _, __ = np.linalg.svd(sigma_)
    return u_[:, :k]


# Plot the corresponding principal components
def plot_data(x, u, mu_):
    """
    :param x:
    :param u:
    :param mu_:
    :return:
    """
    plt.scatter(x[:, 0], x[:, 1], marker='o', color='b')
    for i in range(u.shape[1]):
        plt.plot([mu_[0], mu_[0] + u[0, i]], [mu_[1], mu_[1] + u[1, i]], color='r')
    plt.title('Principal components')
    plt.xlabel('First feature')
    plt.ylabel('Second feature')
    plt.grid()
    plt.show()


#  After computing the principal components, you can use them to reduce the
#  feature dimension of your dataset by projecting each example onto a lower
#  dimensional space, X_reduced = X * U
def project_data(x, u, k):
    """
    投影数据
    :param x:
    :param u:
    :param k:
    :return:
    """
    return x.dot(u[:, :k])


# You should project each example in X onto the
#  top K components in U.
def recover_data(z, u, k):
    """
    恢复数据
    :param z:
    :param u:
    :param k:
    :return:
    """
    return z.dot(u[:, :k].T)


# Visualizing the projections
def plot_projection(x_norm, x_rec):
    """
    绘制投影数据
    :param x_norm:
    :param x_rec:
    :return:
    """
    plt.scatter(x_norm[:, 0], x_norm[:, 1], marker='o', color='b', label='Original data')
    plt.scatter(x_rec[:, 0], x_rec[:, 1], marker='o', color='r', label='Projected data')
    plt.title('The Normalized and Projected Data after PCA')
    plt.xlabel('First feature')
    plt.ylabel('Second feature')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    X = np.array(data_pca)
    # Normalize the data
    X_norm, mu, sigma = feature_normalize(X)
    # Run PCA
    U = pca(X_norm, 2)
    print('Top principal component is ', U[:, 0])
    # Plot the data with principal components
    plot_data(X, U, mu)
    # Project the data
    K = 1
    Z = project_data(X_norm, U, K)
    print('Projection of the first example is ', Z[0])
    # Recover the data
    X_rec = recover_data(Z, U, K)
    print('Recovery of the first example is ', X_rec[0])
    # Plot the normalized and projected data
    plot_projection(X_norm, X_rec)
