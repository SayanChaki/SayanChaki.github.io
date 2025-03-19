import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer

# Generate sample data (we know there are 4 clusters, but we'll pretend we don't)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# Method 1: Elbow Method
# Try different numbers of clusters and calculate inertia (sum of squared distances)
inertias = []
silhouette_scores = []
range_of_k = range(1, 11)  # Try k from 1 to 10

for k in range_of_k:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    
    # Can't calculate silhouette score for k=1
    if k > 1:
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    else:
        silhouette_scores.append(0)

# Plot the Elbow Method results
plt.figure(figsize=(12, 5))

# Inertia plot
plt.subplot(1, 2, 1)
plt.plot(range_of_k, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)



# Print the optimal number of clusters based on different methods
print("Optimal number of clusters based on different methods:")

