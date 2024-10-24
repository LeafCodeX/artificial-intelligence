from k_means import k_means
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_iris():
    data = pd.read_csv("data/iris.data", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    #print(data)
    classes = data["class"].to_numpy()
    features = data.drop("class", axis=1).to_numpy()
    return features, classes


def evaluate(clusters, labels):
    for cluster in np.unique(clusters):
        labels_in_cluster = labels[clusters==cluster]
        print(f"Cluster: {cluster}")
        for label_type in np.unique(labels):
            print(f"Num of {label_type}: {np.sum(labels_in_cluster==label_type)}")

def clustering(kmeans_pp):
    data = load_iris()
    features, classes = data
    intra_class_variance = []
    for i in range(100):
        assignments, centroids, error = k_means(features, 3, kmeans_pp)
        #evaluate(assignments, classes)
        intra_class_variance.append(error)
    print(f"Mean intra-class variance: {np.mean(intra_class_variance)}")

    # Plot from: https://www.kaggle.com/code/khotijahs1/k-means-clustering-of-iris-dataset
    # Plotting the clusters
    fig, ax = plt.subplots()
    colors = ['purple', 'orange', 'green']
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    for i in range(3):
        ax.scatter(features[assignments == i, 0], features[assignments == i, 1], s=100, c=colors[i], label=labels[i])
    # Plotting the centroids of the clusters
    ax.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', label='Centroids')
    ax.grid(True)
    ax.set_axisbelow(True)  # This line makes the grid lines go behind other graph elements
    ax.legend()
    plt.show()


if __name__=="__main__":
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("[INFO] Clustering with KMeans++ initialization:")
    clustering(kmeans_pp=True)
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    print("[INFO] Clustering with Forgy (random) initialization:")
    clustering(kmeans_pp=False)
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")




