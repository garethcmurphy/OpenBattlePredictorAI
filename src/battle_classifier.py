#!/usr/bin/env python3
"""Script to classify battles"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class BattleClassifier:
    """Class to classify battles"""
    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Load data from CSV"""
        self.data = pd.read_csv(self.data_path)
        self.X = self.data.drop(columns=["Outcome"])
        self.y = self.data["Outcome"]

    def split_data(self):
        """Split data into training and testing sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )

    def train_model(self):
        """Train the model"""
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluate the model"""
        y_pred = self.model.predict(self.X_test)
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

    def plot_feature_importance(self, output_path):
        """Plot feature importance"""
        importances = self.model.feature_importances_
        feature_names = self.X.columns
        plt.barh(feature_names, importances)
        plt.title("Feature Importance")
        plt.savefig(output_path)
        plt.show()

if __name__ == "__main__":
    classifier = BattleClassifier(data_path="../results/battles_with_clusters.csv")
    classifier.load_data()
    classifier.split_data()
    classifier.train_model()
    classifier.evaluate_model()
    classifier.plot_feature_importance(output_path="../results/feature_importance.png")
