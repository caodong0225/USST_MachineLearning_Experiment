from dataset import data_kmeans
import matplotlib.pyplot as plt
import numpy as np


# Finding close centroids
def find_closest_centroids(X, centroids):
    idx = []
    for x in X:
        idx.append(min(range(len(centroids)), key=lambda i: np.sum((np.array(x) - np.array(centroids[i])) ** 2)))
    return idx


# Computing centroid means
def compute_centroids(X, idx, k):
    m, n = len(X), len(X[0])
    centroids = []
    for i in range(k):
        sum_ = np.zeros(n)
        count = 0
        for j in range(m):
            if idx[j] == i:
                sum_ += np.array(X[j])
                count += 1
        centroids.append((sum_ / count).tolist())
    return centroids


# train
def train(X, k, max_iters):
    m, n = len(X), len(X[0])
    centroids = X[:k]
    centroids_list = []
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)
        centroids_list.append(centroids)
    return centroids_list


# 可视化,标注数据点的移动轨迹
def plot_data(X, idx, centroids_list):
    # 已知一共有3个类别
    colors = ['r', 'g', 'b']
    # 画出数据点
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], color=colors[idx[i]])
    # 用折线图表示中心点的移动轨迹，在一张图中画出来
    # 画出中心点的移动轨迹
    # 用不同的颜色表示不同的中心点
    for i in range(len(centroids_list[0])):
        x = [centroids[i][0] for centroids in centroids_list]
        y = [centroids[i][1] for centroids in centroids_list]
        plt.plot(x, y, color=colors[i])
        # 画出中心点
        # 用黑色的x表示中心点
        for centroids in centroids_list:
            plt.scatter(centroids[i][0], centroids[i][1], color='black', marker='x')
    plt.show()


if __name__ == "__main__":
    k = 3
    max_iters = 10
    centroids_list = train(data_kmeans, k, max_iters)
    centroids = centroids_list[-1]
    idx = find_closest_centroids(data_kmeans, centroids)
    plot_data(data_kmeans, idx, centroids_list)
