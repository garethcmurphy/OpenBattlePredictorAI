#!/usr/bin/env python3
"""
machine learning to win napoleon's battles
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create Sample Dataset
data = {
    "Battle Name": ["Austerlitz", "Waterloo", "Borodino"],
    "Date": ["1805-12-02", "1815-06-18", "1812-09-07"],
    "Commander": ["Napoleon", "Napoleon", "Napoleon"],
    "Troop Strength": [67000, 73000, 130000],
    "Opponent Commander": ["Tsar Alexander I", "Duke of Wellington", "Mikhail Kutuzov"],
    "Opponent Strength": [85000, 113000, 120000],
    "Terrain": ["Hills, Lakes", "Flat Plains", "Flat, River Nearby"],
    "Weather": ["Foggy", "Rainy", "Clear"],
    "Strategy Used": ["Flanking, Divide and Conquer", "Direct Assault", "Artillery, Attrition"],
    "Outcome": ["Victory", "Defeat", "Stalemate"]
}
df = pd.DataFrame(data)

# Step 2: Preprocess Data
label_encoder = LabelEncoder()
df['Outcome'] = label_encoder.fit_transform(df['Outcome'])  # Encode outcome
df['Terrain'] = label_encoder.fit_transform(df['Terrain'])  # Encode terrain
df['Weather'] = label_encoder.fit_transform(df['Weather'])  # Encode weather

# Add a derived feature
df['Troop Ratio'] = df['Troop Strength'] / df['Opponent Strength']

# Select Features and Labels
X = df[['Troop Strength', 'Opponent Strength', 'Terrain', 'Weather', 'Troop Ratio']]
y = df['Outcome']

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
