import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import mean_squared_error

# Generate synthetic Tic-Tac-Toe states data (flattened 3x3 board states as a 9-feature dataset)
def generate_tictactoe_data(n_samples=100):
    data = []
    for _ in range(n_samples):
        board = np.random.choice([-1, 0, 1], size=(3, 3))  # -1 for O, 1 for X, 0 for empty
        data.append(board.flatten())
    return np.array(data)

# Generate Tic-Tac-Toe states
X = generate_tictactoe_data()

# Function to visualize clusters
def plot_clusters(X, labels, title):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

### 1. K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
initial_kmeans_labels = np.random.randint(0, 3, size=len(X))  # Random initial clusters

print("K-Means Clustering")
print("1. Initial clusters:")
plot_clusters(X, initial_kmeans_labels, "Initial K-Means Clusters")

# Fitting K-Means and finding final clusters
kmeans.fit(X)
kmeans_labels = kmeans.labels_

# Calculating error rate for K-Means
centroids_kmeans = kmeans.cluster_centers_
error_rate_kmeans = mean_squared_error(X, centroids_kmeans[kmeans_labels])

print("2. Final clusters with epoch size (Iterations):", kmeans.n_iter_)
plot_clusters(X, kmeans_labels, "Final K-Means Clusters")

print(f"3. Final clusters with error rate: {error_rate_kmeans:.2f}\n")

### 2. Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
initial_agg_labels = np.random.randint(0, 3, size=len(X))  # Random initial clusters

print("\nAgglomerative Clustering")
print("1. Initial clusters:")
plot_clusters(X, initial_agg_labels, "Initial Agglomerative Clusters")

# Fitting Agglomerative Clustering and finding final clusters
agg_labels = agg.fit_predict(X)

# Calculating error rate for Agglomerative Clustering
unique_labels = np.unique(agg_labels)
centroids_agg = np.array([X[agg_labels == label].mean(axis=0) for label in unique_labels])
error_rate_agg = mean_squared_error(X, centroids_agg[agg_labels])

print("2. Final clusters (Agglomerative does not have epochs)")
plot_clusters(X, agg_labels, "Final Agglomerative Clusters")

print(f"3. Final clusters with error rate: {error_rate_agg:.2f}")
