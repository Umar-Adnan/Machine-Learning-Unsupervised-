import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------------------------------------------------------
# 1. GENERATE SYNTHETIC FINTECH DATA (Simulating 1,000 Wallet Users)
# -----------------------------------------------------------------------------
np.random.seed(42)

# Simulating 3 distinct underlying behaviors
# Persona 1: "The Bill Payers" (Low P2P, High Bill Pay, Medium Value, High Activity)
p1 = np.random.normal(loc=[5, 25, 5000, 20], scale=[2, 5, 1000, 4], size=(300, 4))
# Persona 2: "The Heavy Transferors" (High P2P, Low Bill Pay, High Value, Medium Activity)
p2 = np.random.normal(loc=[30, 2, 25000, 10], scale=[8, 1, 5000, 3], size=(400, 4))
# Persona 3: "The Inactive/Micro Users" (Low P2P, Low Bill Pay, Low Value, Low Activity)
p3 = np.random.normal(loc=[2, 1, 800, 3], scale=[1, 1, 200, 1], size=(300, 4))

# Combine and ensure no negative values
data = np.vstack([p1, p2, p3])
data = np.abs(data) 

# Create DataFrame
columns = ['p2p_transactions', 'bill_payments', 'avg_transaction_value', 'active_days']
df = pd.DataFrame(data, columns=columns)

print("=== Raw Dataset Sample ===")
print(df.head(), "\n")

# -----------------------------------------------------------------------------
# 2. DATA PREPROCESSING & SCALING
# -----------------------------------------------------------------------------
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# -----------------------------------------------------------------------------
# 3. DIMENSIONALITY REDUCTION (PCA)
# -----------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
df_pca = pca.fit_transform(df_scaled)

explained_variance = pca.explained_variance_ratio_.sum() * 100
print(f"PCA Compression: Retained {explained_variance:.2f}% of original user behavior variance.\n")

# -----------------------------------------------------------------------------
# 4. K-MEANS CLUSTERING
# -----------------------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df_pca)
centroids = kmeans.cluster_centers_

# -----------------------------------------------------------------------------
# 5. MODEL EVALUATION
# -----------------------------------------------------------------------------
sil_score = silhouette_score(df_pca, df['cluster'])
print("=== Model Evaluation ===")
print(f"Silhouette Score: {sil_score:.3f}")
print("(Scale: -1.0 to 1.0. Higher is better, indicating dense and well-separated clusters.)\n")

# -----------------------------------------------------------------------------
# 6. VISUALIZATION & EXPORT DASHBOARD
# -----------------------------------------------------------------------------
os.makedirs('outputs', exist_ok=True)

plt.figure(figsize=(10, 7))
sns.set_theme(style="darkgrid")

sns.scatterplot(
    x=df_pca[:, 0], y=df_pca[:, 1], 
    hue=df['cluster'], palette='deep', alpha=0.8, s=60, edgecolor='k'
)

plt.scatter(
    centroids[:, 0], centroids[:, 1], 
    c='red', s=300, marker='X', linewidths=2, edgecolors='black', label='Centroids'
)

plt.title('Digital Wallet User Segmentation (PCA + K-Means)', fontsize=16, pad=15)
plt.xlabel('Principal Component 1 (Transaction Volume & Value)')
plt.ylabel('Principal Component 2 (App Activity & Bill Payments)')
plt.legend(title='Discovered Persona')
plt.tight_layout()

export_path = 'wallet_segmentation_clusters.png'
plt.savefig(export_path, dpi=300, bbox_inches='tight')
print(f"Visualization successfully saved to: {export_path}\n")

plt.show()

print("=== Cluster Archetypes (Average Values) ===")
print(df.groupby('cluster').mean().round(2))