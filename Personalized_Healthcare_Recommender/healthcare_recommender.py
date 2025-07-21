# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Step 2: Load the dataset
data = pd.read_csv("healthcare_data.csv")  # make sure the CSV file is in the same folder
print("First 5 rows of data:")
print(data.head())

# Step 3: Define features (X) and target (y)
X = data.drop("Class", axis=1)
y = data["Class"]

# Step 4: Define numerical features (your dataset has no categorical columns)
numerical_features = ['Recency', 'Frequency', 'Monetary', 'Time']

# Step 5: Create preprocessing pipeline
numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features)
])

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build full pipeline with model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Step 8: Train the model
model_pipeline.fit(X_train, y_train)

# Step 9: Evaluate the model
y_pred = model_pipeline.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Personalized Recommendation Function
def generate_recommendation(patient_data):
    prediction = model_pipeline.predict(patient_data)
    recommendation_mapping = {
        0: 'Healthy – No action needed',
        1: 'Needs attention – Recommend follow-up'
    }
    return recommendation_mapping.get(prediction[0], "Unknown")

# Step 11: Example patient input
example_patient = pd.DataFrame({
    'Recency': [1],
    'Frequency': [24],
    'Monetary': [70],
    'Time': [85]
})

# Step 12: Generate recommendation
print("\nGenerated Recommendation for Example Patient:")
print(generate_recommendation(example_patient))

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Class distribution
plt.figure(figsize=(5, 4))
sns.countplot(x='Class', data=data, palette='Set2')
plt.title("Distribution of Classes (Health Status)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks([0, 1], ['Healthy (0)', 'Needs Attention (1)'])
plt.tight_layout()
plt.show()

# 2. Distribution of features
features_to_plot = ['Recency', 'Frequency', 'Monetary', 'Time']
for feature in features_to_plot:
    plt.figure(figsize=(5, 4))
    sns.histplot(data[feature], kde=True, bins=20, color='skyblue')
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# 3. Confusion Matrix (Graphical)
from sklearn.metrics import ConfusionMatrixDisplay

plt.figure(figsize=(6, 5))
cm_display = ConfusionMatrixDisplay.from_estimator(model_pipeline, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
