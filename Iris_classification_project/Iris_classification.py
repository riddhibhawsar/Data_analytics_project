# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Step 2: Load dataset
df = pd.read_csv("Iris.csv")

# Step 3: Basic info
print("\nFirst 5 rows:\n", df.head())
print("\nDataset info:\n")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Step 4: Exploratory Data Analysis (EDA)
sns.pairplot(df, hue='Species')
plt.show()

# Histograms
df.hist(edgecolor='black', linewidth=1.2, figsize=(12, 6))
plt.suptitle("Histogram for each feature", fontsize=16)
plt.show()

# Boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, orient="h")
plt.title("Boxplot of features")
plt.show()

# Step 5: Feature selection
X = df.drop("Species", axis=1)
y = df["Species"]

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Model training (KNN example)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 9: Prediction & Evaluation
y_pred = model.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 10: Confusion matrix heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Heatmap')
plt.show()
