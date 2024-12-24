#!/usr/bin/env python3
"""Clustering battles using K-Means"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class BattleClustering:
    """Clustering battles using K-Means"""
    def __init__(self, data_path, n_clusters=3, random_state=42):
        self.data_path = data_path
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.data = None
        self.data_encoded = None
        self.X_scaled = None
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.clusters = None

    def load_data(self):
        """Load dataset from the specified path"""
        self.data = pd.read_csv(self.data_path)

    def preprocess_data(self):
        """Preprocess data by encoding categorical columns and scaling features"""
        categorical_columns = ["Terrain", "Weather", "Key_Factors"]
        self.data_encoded = pd.get_dummies(
            self.data, columns=categorical_columns, drop_first=True
        )
        X = self.data_encoded.drop(
            columns=[
                "Outcome",
                "Battle_Name",
            ]
        )  # Drop the target and non-feature columns
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)

    def perform_clustering(self):
        """Perform K-Means clustering on the preprocessed data"""
        self.clusters = self.kmeans.fit_predict(self.X_scaled)
        self.data["Cluster"] = self.clusters

    def save_results(self, output_path):
        """Save the clustering results to a CSV file"""
        self.data.to_csv(output_path, index=False)

    def visualize_clusters(self, output_path):
        """Visualize the clusters (if dataset is 2D or reduced to 2D)"""
        plt.scatter(
            self.X_scaled[:, 0], self.X_scaled[:, 1], c=self.clusters, cmap="viridis"
        )
        plt.title("Battle Clusters")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.savefig(output_path)


if __name__ == "__main__":
    clustering = BattleClustering(data_path="data/napoleon_battles.csv")
    clustering.load_data()
    clustering.preprocess_data()
    clustering.perform_clustering()
    clustering.save_results(output_path="results/battles_with_clusters.csv")
    clustering.visualize_clusters(output_path="results/clusters_visualization.png")
