"""
Diabetes Prediction Model Training Script
Trains a RandomForestClassifier on the Pima Indians Diabetes Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the Pima Indians Diabetes Dataset
# The dataset is loaded from a well-known source
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

print("Loading dataset...")
try:
    df = pd.read_csv(url, names=column_names)
    print(f"Dataset loaded successfully! Shape: {df.shape}")
except Exception as e:
    print(f"Error loading from URL: {e}")
    print("Please ensure you have internet connection or download the dataset manually.")
    exit(1)

# Display basic information about the dataset
print("\nDataset Info:")
print(df.head())
print(f"\nMissing values:\n{df.isnull().sum()}")

# Handle missing values - In the Pima Indians dataset, 0 values are often used to represent missing data
# for certain features like Glucose, BloodPressure, SkinThickness, Insulin, BMI
features_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for feature in features_to_check:
    # Replace 0 values with NaN (as 0 is not a valid value for these features)
    df[feature] = df[feature].replace(0, np.nan)

# Fill missing values with median
print("\nFilling missing values with median...")
for feature in features_to_check:
    median_value = df[feature].median()
    df[feature].fillna(median_value, inplace=True)

print(f"\nMissing values after imputation:\n{df.isnull().sum()}")

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets (80/20)
print("\nSplitting data into train (80%) and test (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train RandomForestClassifier
print("\nTraining RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Diabetic', 'Diabetic']))

# Save the model and scaler
print("\nSaving model and scaler...")
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model saved as 'diabetes_model.pkl'")
print("Scaler saved as 'scaler.pkl'")
print("\nTraining completed successfully!")

