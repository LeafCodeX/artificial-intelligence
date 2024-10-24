import numpy as np

def initialize_centroids_forgy(data, num_clusters):
    # TODO implement random initialization
    random_indices = np.random.choice(np.arange(data.shape[0]), size=num_clusters, replace=False)
    return data[random_indices, :]

"""def initialize_centroids_kmeans_pp(data, num_clusters):
    # TODO implement kmeans++ initizalization
    centroids = np.empty((num_clusters, data.shape[1]))
    first_random_index = np.random.randint(data.shape[0])
    centroids[0, :] = data[first_random_index, :]

    for cluster_index in np.arange(1, num_clusters):
        # squared_distances = np.sum((data.reshape((data.shape[0], 1, data.shape[1])) - centroids[:cluster_index, :].reshape((1, cluster_index, centroids.shape[1])))**2, axis=-1)
        squared_difference = np.empty((data.shape[0], cluster_index, data.shape[1]))

        for i in range(data.shape[0]):
            for j in range(cluster_index):
                difference = data[i, :] - centroids[j, :]
                squared_difference[i, j, :] = difference ** 2

        max_squared_distances = np.max(squared_difference, axis=1)
        normalized_distances = max_squared_distances / np.max(max_squared_distances)
        index_of_closest_data_point = np.argmin(normalized_distances)
        centroids[cluster_index, :] = data[index_of_closest_data_point, :]

    return centroids"""

def initialize_centroids_kmeans_pp(data, num_clusters):
    # TODO implement kmeans++ initizalization
    cluster_centers = np.zeros((num_clusters, data.shape[1]))
    cluster_centers[0] = data[np.random.choice(data.shape[0], 1, replace=False)]

    for cluster_index in range(1, num_clusters):
        squared_distances = np.zeros(data.shape[0])
        for data_index in range(data.shape[0]):
            dist = np.inf
            for current_center in cluster_centers[:cluster_index]:
                loc_dist = np.sqrt(np.sum((data[data_index] - current_center) ** 2))
                if loc_dist<dist:
                    dist=loc_dist
            squared_distances[data_index] = dist
        cluster_centers[cluster_index] = data[squared_distances.argmax()]

    return cluster_centers

def assign_to_cluster(data, centroids):
    # TODO find the closest cluster for each data point
    squared_distances_to_centroids = np.sum((data.reshape((data.shape[0], 1, data.shape[1])) - centroids.reshape((1, *centroids.shape)))**2, axis=-1)
    cluster_assignments: np.ndarray = np.argmin(squared_distances_to_centroids, axis=-1)
    return cluster_assignments

def update_centroids(data, cluster_assignments, num_centroids):
    # TODO find new centroids based on the assignments
    updated_centroids = np.empty((num_centroids, data.shape[1]))
    for cluster_id in np.arange(num_centroids):
        updated_centroids[cluster_id, :] = np.mean(data[cluster_assignments == cluster_id], axis=0)
    return updated_centroids

def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        # print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments, num_centroids)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

