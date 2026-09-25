# Digital Wallet User Segmentation: An Unsupervised Learning Pipeline

This repository serves as a core AI/ML portfolio piece for Semester 5, demonstrating the end-to-end architecture of an unsupervised machine learning pipeline. It transitions from foundational mathematical concepts to a production-grade business application: autonomously segmenting peer-to-peer (P2P) digital wallet users based on their transaction behaviors.

---

## 🧠 Part 1: Unsupervised Learning Foundations

Unlike supervised learning, which relies on predefined labels ($y$) to train a model, unsupervised learning relies entirely on the feature matrix ($X$) to autonomously discover hidden patterns, groupings, and anomalies in raw data. 

This repository implements the following core concepts:

*   **Euclidean Distance & Feature Scaling:** The fundamental measure of similarity. Because distance metrics are highly sensitive to unscaled data (e.g., transaction volumes in the tens vs. monetary values in the thousands), `StandardScaler` is enforced to normalize variance.
*   **Variance as Information:** Algorithms hunt for spread. Features with high variance carry the most informational signal for clustering.
*   **Principal Component Analysis (PCA):** A dimensionality reduction technique that solves the "Curse of Dimensionality." It compresses high-dimensional data into fewer principal components (PC1, PC2) by maximizing variance, stripping redundant noise, and enabling 2D visualization.
*   **K-Means Clustering:** A hard-clustering algorithm that partitions data into $k$ distinct, non-overlapping groups by iteratively shifting multi-dimensional centroids to minimize inertia (Within-Cluster Sum of Squares).
*   **The Elbow Method:** A diagnostic heuristic used to determine the mathematically optimal number of clusters ($k$) by plotting inertia against $k$ values and identifying the point of diminishing returns.
*   **Silhouette Score:** The primary evaluation metric for unlabelled data. It validates cluster cohesion (tightness) and separation (distance from neighboring clusters) on a scale of -1.0 to 1.0.

---

## 🚀 Part 2: The Project Architecture

### The Business Problem
Fintech applications and digital money institutions process millions of daily transactions. To optimize marketing, user onboarding, and feature rollouts, businesses must understand distinct user personas. This project autonomously identifies these personas without human bias.

### The Dataset
A synthetic dataset of 1,000 digital wallet users was generated with four core behavioral features tracked over a 30-day window:
1.  **`p2p_transactions`**: Frequency of peer-to-peer transfers.
2.  **`bill_payments`**: Frequency of utility/merchant payments.
3.  **`avg_transaction_value`**: The monetary volume of the transactions.
4.  **`active_days`**: Platform engagement rate.

### The Pipeline Execution
1.  **Preprocessing:** Applied `StandardScaler` to level the mathematical playing field between frequency counts and financial values.
2.  **Compression:** Executed `PCA` to reduce the 4-dimensional data into 2 Principal Components, retaining the maximum possible behavioral variance.
3.  **Clustering:** Deployed `K-Means` with $k=3$ (validated via Elbow Method logic) to group the PCA-reduced data.
4.  **Evaluation:** Validated model accuracy mathematically using the `Silhouette Score`.
5.  **Visualization:** Generated a high-resolution, presentation-ready scatter plot mapping the distinct user clusters and their central archetypes.

### Discovered Personas (Cluster Archetypes)
By reverse-engineering the cluster centroids back to the original data, the model successfully identified three distinct financial behaviors:

*   **Cluster 0 | The Heavy Transferors:** High P2P transaction frequency and massive average transaction values. These are power users moving significant capital.
*   **Cluster 1 | The Utility Payers:** Low P2P transfers but high bill payment frequency and high daily app engagement. These users rely on the app for daily financial management.
*   **Cluster 2 | The Dormant Accounts:** Low transaction frequency, minimal financial value, and low platform engagement. These represent churn risks or inactive micro-users.

