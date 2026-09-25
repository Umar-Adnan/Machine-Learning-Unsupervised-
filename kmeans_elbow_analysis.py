import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1. GENERATE SAMPLE DATA
# We create 500 unlabelled data points naturally grouped into 4 hidden clusters.
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.5, random_state=42)

# 2. THE ELBOW METHOD
inertias = []
k_values = range(1, 11)

# Loop through k=1 to k=10, fitting a new K-Means model each time
for k in k_values:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(X)
    inertias.append(kmeans_test.inertia_)

# 3. EXECUTE FINAL K-MEANS (Assuming the elbow shows k=4)
kmeans_final = KMeans(n_clusters=4, random_state=42, n_init=10)
y_kmeans = kmeans_final.fit_predict(X)
centroids = kmeans_final.cluster_centers_

# 4. VISUALIZATION DASHBOARD
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: The Elbow Graph
axes[0].plot(k_values, inertias, marker='o', linestyle='--', color='b')
axes[0].set_title('The Elbow Method: Finding Optimal k')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (Within-Cluster Variance)')
axes[0].annotate('The "Elbow" (k=4)', xy=(4, inertias[3]), xytext=(5, inertias[3] + 2000),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

# Plot B: The Final K-Means Clusters
# Scatter plot the data points, coloring them by their assigned cluster
axes[1].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6, edgecolor='k')
# Overlay the final centroids in red
axes[1].scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
axes[1].set_title('Final K-Means Clustering (k=4)')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()

plt.tight_layout()
plt.show()