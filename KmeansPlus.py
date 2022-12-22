import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
algorithm logic

this algorithm will implement mathematical logic following step
1.read dataset
2.set clusters, and randomly set it's centroid
3.process clustering using eculidian distance every centroid will judge it's distnace from each dataset
4.update centroid using kmenas++ algorithm algorithm is as follow

4-1.Optionally specify the first centroid.

4-2.For each remaining data point, calculate the distance to the nearest centroids.

4-3.The following centroid is specified based on the probability
    that the distance between each data point and the nearest centroid is proportional.
    This prevents the next centroid from approaching the already specified centroid.

4-4Repeat steps 2 through 3 until you specify all k centroids.
"""
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def pca(X, num_components):
  # Mean center the data
  X_mean = X - np.mean(X, axis=0)

  # Calculate the covariance matrix
  cov = np.cov(X_mean, rowvar=False)

  # Calculate the eigenvalues and eigenvectors of the covariance matrix
  eigenvalues, eigenvectors = np.linalg.eig(cov)

  # Sort the eigenvalues and eigenvectors in descending order
  idx = eigenvalues.argsort()[::-1]
  eigenvalues = eigenvalues[idx]
  eigenvectors = eigenvectors[:,idx]

  # Take the top `num_components` eigenvectors
  W = eigenvectors[:, :num_components]

  # Project the data onto the new subspace
  X_pca = np.dot(X_mean, W)

  return X_pca
class KMeans:

    def __init__(self, K=3, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # the centers (mean vector) for each cluster
        self.centroids = [[] for _ in range(self.K)]


    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape #sample 1500 features 600

        """initialize using Kmean method"""

        # random_centroid_position = np.random.choice(self.n_samples, self.K, replace=False)
        # self.centroids = [self.X[idx] for idx in random_centroid_position]
    
    
        """initialize using Kmeans++ method"""
        random_centroid_position = np.random.choice(self.n_samples, 1, replace=False)
        self.centroids = np.empty((self.K, self.n_features))#number of cluster and
        self.centroids[0] = X[random_centroid_position]
        for c_id in range(1,self.K):

            ## initialize a list to store distances of data
            ## points from nearest centroid
            dist = []
            for i in range(X.shape[0]):
                point = X[i, :]
                d = float("inf")

                ## compute distance of 'point' from each of the previously
                ## selected centroid and store the minimum distance
                for j in range(self.K):
                    temp_dist = euclidean_distance(point, self.centroids[j])
                    d = min(d, temp_dist)
                dist.append(d)

            ## select data point with maximum distance as our next centroid
            dist = np.array(dist)
            next_centroid = X[np.argmax(dist), :]
            self.centroids[c_id]= next_centroid
            dist = []
        # return self.centroids


        #print (self.centroids)

        # optimize clusters
        for _ in range(20):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            #print (self.centroids)

            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classify samples as the index of their clusters
        # return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels


    def _create_clusters(self, centroids):
        """
        assign the samples to the closest centroids

        logic for the project
        1.we have 1500 or 3000 sample and 600 features of data
        2.we will pick nearest dataset from sample data for each index
        3.append the index to nearest centroid
        """
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        """
        distance of the current sample to each centroid

        1.take centroids we initialized
        2.distances will get K datasets (eculidean distance) for each data
        3.pick nearest one
        """
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx


    def _get_centroids(self, clusters):
        """
        assign mean value of clusters to centroids

        from logic create_centroid we assigned clusters for each centroids
        we will calculate mean value of clusters and allocate it new centroids
        """
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)#각 그룹의 같은 원소들
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        """
        distances between old and new centroids, for all centroids
        if sumof distances equals to 0, which means centroid coverged
        """
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="*", color="black", linewidth=2)

        for point in self.centroids:
            ax.scatter(*point, marker="*", color="black", linewidth=2)

        plt.show()


# Testing
if __name__ == "__main__":
    # np.random.seed(42)
    # from sklearn.datasets import make_blobs

    # X, y = make_blobs(
    #     centers=3, n_samples=500, n_features=2, shuffle=True, random_state=42
    # )
    # print(X.shape)

    # clusters = len(np.unique(y))
    # print(clusters)
    X = pd.read_csv('.\\data_for_student\\train\\data.csv')
    y = pd.read_csv('.\\data_for_student\\train\\data.csv')

    X = X.transpose()
    index = [i for i in range (len(X))]
    X.index = index

    y=y.transpose()
    vector = np.vectorize(np.int_)
    y_data = y.index.values.astype(float)
    y_data = vector((y_data))
    y = pd.Series(y_data)

    X = X.to_numpy()
    y= y.to_numpy()
    _X = pca_compoments = pca(X,3)
    _X = _X[:,:2]
    clusters = len(np.unique(y))

    print(_X.shape,'\n',y.shape)
    k = KMeans(K=clusters, max_iters=150, plot_steps=False)
    k.predict(_X)
    y_pred = k.centroids
    y_pred = k._get_cluster_labels(clusters)
    k.plot()

