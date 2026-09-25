import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------
# 1. GENERATE HIGH-DIMENSIONAL DATA
# -----------------------------------------------------------------------------
# Creating 500 samples with 10 features (too complex for a standard scatter plot)
X, _ = make_blobs(n_samples=500, n_features=10, centers=4, cluster_std=2.0, random_state=42)

# -----------------------------------------------------------------------------
# 2. PREPROCESSING: Feature Scaling
# -----------------------------------------------------------------------------
# Crucial: PCA and K-Means are distance/variance based. We MUST scale the data.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------------------------
# 3. DIMENSIONALITY REDUCTION: PCA
# -----------------------------------------------------------------------------
# Compress the 10 features down to just 2 Principal Components
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Check how much information (variance) we retained after dropping 8 features
explained_variance = pca.explained_variance_ratio_.sum() * 100
print(f"Information Retained: {explained_variance:.2f}%")

# -----------------------------------------------------------------------------
# 4. K-MEANS CLUSTERING (Execution)
# -----------------------------------------------------------------------------
# We run K-Means on the compressed 2D data, not the original 10D data
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)
centroids = kmeans.cluster_centers_

# -----------------------------------------------------------------------------
# 5. VISUALIZATION DASHBOARD
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 7))

# Plot the PCA-compressed data, colored by their K-Means cluster assignment
sns.scatterplot(
    x=X_pca[:, 0], 
    y=X_pca[:, 1], 
    hue=cluster_labels, 
    palette='viridis', 
    alpha=0.7, 
    edgecolor='k',
    legend='full'
)

# Overlay the cluster centroids
plt.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    c='red', 
    s=250, 
    marker='X', 
    label='Centroids'
)

plt.title('Unsupervised Pipeline: PCA Compression + K-Means Clustering')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()