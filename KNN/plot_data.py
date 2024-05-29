from dataset import data_kmeans
import matplotlib.pyplot as plt


def show_data(data):
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    plt.scatter(x, y)
    plt.show()


if __name__ == "__main__":
    show_data(data_kmeans)
