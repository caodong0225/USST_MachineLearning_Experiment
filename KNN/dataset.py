with open("ex7data2.txt", "r") as f:
    data_kmeans = f.readlines()
data_kmeans = [list(map(float, x.strip().split('\t'))) for x in data_kmeans]
