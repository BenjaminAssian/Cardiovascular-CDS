import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("../data/heart.csv")

print("Dataset Preview:")
print(df.head())

# -------------------------
# DATA CLEANING
# -------------------------

# Convert categorical columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Remove missing values
df = df.dropna()

print("\nCleaned dataset shape:", df.shape)

# -------------------------
# EXPLORATORY DATA ANALYSIS
# -------------------------

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("../visualizations/correlation_heatmap.png")

plt.figure()
df["Age"].hist()
plt.title("Age Distribution")
plt.savefig("../visualizations/age_distribution.png")

# FEATURE SELECTION

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# -------------------------
# TRAIN TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MODEL TRAINING
# -------------------------

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -------------------------
# MODEL EVALUATION
# -------------------------

print("\n--- Model Evaluation ---")

# Predictions
predictions = model.predict(X_test)

# Dynamic accuracy calculation
accuracy = accuracy_score(y_test, predictions)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Classification metrics
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("../visualizations/confusion_matrix.png")

# -------------------------
# CROSS VALIDATION (NEW)
# -------------------------

scores = cross_val_score(model, X, y, cv=5)

print("\nCross Validation Scores:", scores)
print(f"Average Cross-Validated Accuracy: {scores.mean() * 100:.2f}%")

# -------------------------
# SAVE MODEL
# -------------------------

joblib.dump(model, "../model/heart_model.pkl")

print("\nModel saved to ../model/heart_model.pkl")