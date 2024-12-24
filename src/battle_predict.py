#!/usr/bin/env python3
"""
Machine learning to win Napoleon's battles
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


class NapoleonicBattlePredictor:
    """
    Machine learning to win Napoleon's battles
    """

    def __init__(self):
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        data = {
            "Battle Name": ["Austerlitz", "Waterloo", "Borodino"],
            "Date": ["1805-12-02", "1815-06-18", "1812-09-07"],
            "Commander": ["Napoleon", "Napoleon", "Napoleon"],
            "Troop Strength": [67000, 73000, 130000],
            "Opponent Commander": [
                "Tsar Alexander I",
                "Duke of Wellington",
                "Mikhail Kutuzov",
            ],
            "Opponent Strength": [85000, 113000, 120000],
            "Terrain": ["Hills, Lakes", "Flat Plains", "Flat, River Nearby"],
            "Weather": ["Foggy", "Rainy", "Clear"],
            "Strategy Used": [
                "Flanking, Divide and Conquer",
                "Direct Assault",
                "Artillery, Attrition",
            ],
            "Outcome": ["Victory", "Defeat", "Stalemate"],
        }
        self.data = pd.DataFrame(data)
        self.data.to_csv("battles.csv", index=False)

    def preprocess_data(self):
        self.data["Outcome"] = self.label_encoder.fit_transform(
            self.data["Outcome"]
        )  # Encode outcome
        self.data["Terrain"] = self.label_encoder.fit_transform(
            self.data["Terrain"]
        )  # Encode terrain
        self.data["Weather"] = self.label_encoder.fit_transform(
            self.data["Weather"]
        )  # Encode weather

        # Add a derived feature
        self.data["Troop Ratio"] = (
            self.data["Troop Strength"] / self.data["Opponent Strength"]
        )

        # Select Features and Labels
        self.X = self.data[
            ["Troop Strength", "Opponent Strength", "Terrain", "Weather", "Troop Ratio"]
        ]
        self.y = self.data["Outcome"]

        # Scale the data
        self.X = self.scaler.fit_transform(self.X)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    predictor = NapoleonicBattlePredictor()
    predictor.load_data()
    predictor.preprocess_data()
    predictor.train_model()
