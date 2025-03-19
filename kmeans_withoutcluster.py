import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
# Create 300 points in 4 clusters
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the results
plt.figure(figsize=(10, 6))

# Plot the data points colored by cluster
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title('K-means Clustering with scikit-learn')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Print some information about the clustering
print(f"Number of clusters: {kmeans.n_clusters}")
print(f"Inertia (sum of squared distances to nearest centroid): {kmeans.inertia_:.2f}")
print(f"Number of iterations to converge: {kmeans.n_iter_}")
print("\nCluster centers:")
for i, centroid in enumerate(centroids):
    print(f"Centroid {i+1}: {centroid}")

# Count number of points in each cluster
unique, counts = np.unique(labels, return_counts=True)
print("\nPoints per cluster:")
for i, count in enumerate(counts):
    print(f"Cluster {i+1}: {count} points")
