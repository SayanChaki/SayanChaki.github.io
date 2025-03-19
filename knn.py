import numpy as np
import math

def knn_predict(data, new_point):
    X = np.array([point[0] for point in data])  # Real number part
    y = np.array([point[1] for point in data])  # Label part (1 or 0)

    # Calculate the number of neighbors, K
    n = len(data)
    K = int(math.sqrt(n))

    # Calculate the Euclidean distance between the new point and all data points
    distances = np.abs(X - new_point[0])

    # Get the indices of the K nearest neighbors
    nearest_indices = np.argsort(distances)[:K]

    # Find the labels of the K nearest neighbors
    nearest_labels = y[nearest_indices]

    # Return the majority label (1 or 0) from the nearest neighbors
    predicted_label = np.bincount(nearest_labels).argmax()

    return predicted_label

=data = [(2.5, 1), (1.1, 0), (3.3, 1), (0.7, 0), (4.2, 1)]  # (x, y) where y is label
new_point = (2.0, )  # Only the real number part is used to calculate distance
predicted_label = knn_predict(data, new_point)
print(f'Predicted label for {new_point[0]}: {predicted_label}')

