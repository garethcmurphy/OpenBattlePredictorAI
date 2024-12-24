#!/usr/bin/env python3
"""clustering battles"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("../data/napoleon_battles.csv")  # Adjust path as needed

# Feature selection and scaling
X = data.drop(columns=["Outcome"])  # Drop the target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
data["Cluster"] = clusters

# Save clusters to CSV
data.to_csv("../results/battles_with_clusters.csv", index=False)

# Visualize clusters (if dataset is 2D or reduced to 2D)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap="viridis")
plt.title("Battle Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("../results/clusters_visualization.png")
plt.show()
