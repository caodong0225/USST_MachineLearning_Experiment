"""
Author: caodong0225
Date: 2024-06-04
This file contains the dataset for the KNN algorithm
"""
import matplotlib.pyplot as plt
import numpy as np
from dataset import data_kmeans


# Finding close centroids
def find_closest_centroids(x_, centroids_):
    """
    Find the closest centroids
    :param x_:
    :param centroids_:
    :return:
    """
    idx_ = []
    for x_value in x_:
        idx_.append(min(range(len(centroids_)),
                        key=lambda i, x_v=x_value: np.sum((x_v -
                                                           np.array(centroids_[i])) ** 2)))
    return idx_


# Computing centroid means
def compute_centroids(x_, idx_, k_):
    """
    Compute the centroid means
    :param x_:
    :param idx_:
    :param k_:
    :return:
    """
    m, n = len(x_), len(x_[0])
    centroids_ = []
    for i in range(k_):
        sum_ = np.zeros(n)
        count = 0
        for j in range(m):
            if idx_[j] == i:
                sum_ += np.array(x_[j])
                count += 1
        centroids_.append((sum_ / count).tolist())
    return centroids_


# train
def train(x_, k_, max_iters_):
    """
    Train
    :param x_:
    :param k_:
    :param max_iters_:
    :return:
    """
    centroids_ = x_[:k_]
    centroids_list_ = []
    for _ in range(max_iters_):
        idx_ = find_closest_centroids(x_, centroids_)
        centroids_ = compute_centroids(x_, idx_, k_)
        centroids_list_.append(centroids_)
    return centroids_list_


# 可视化,标注数据点的移动轨迹
def plot_data(x_, idx_, centroids_list_):
    """
    Plot data
    :param x_:
    :param idx_:
    :param centroids_list_:
    :return:
    """
    # 已知一共有3个类别
    colors = ['r', 'g', 'b']
    # 画出数据点
    for index, x_index in enumerate(x_):
        plt.scatter(x_index[0], x_index[1], color=colors[idx_[index]])
    # 用折线图表示中心点的移动轨迹，在一张图中画出来
    # 画出中心点的移动轨迹
    # 用不同的颜色表示不同的中心点
    for index in range(len(centroids_list_[0])):
        x = [centroids_[index][0] for centroids_ in centroids_list_]
        y = [centroids_[index][1] for centroids_ in centroids_list_]
        plt.plot(x, y, color=colors[index])
        # 画出中心点
        # 用黑色的x表示中心点
        for centroids_ in centroids_list_:
            plt.scatter(centroids_[index][0], centroids_[index][1], color='black', marker='x')
    plt.show()


if __name__ == "__main__":
    k = 3
    MAX_ITERS = 10
    centroids_list = train(data_kmeans, k, MAX_ITERS)
    centroids = centroids_list[-1]
    idx = find_closest_centroids(data_kmeans, centroids)
    plot_data(data_kmeans, idx, centroids_list)
