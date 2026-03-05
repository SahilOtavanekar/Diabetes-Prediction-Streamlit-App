"""
Diabetes Prediction Model Training Script
Trains a RandomForestClassifier on the Pima Indians Diabetes Dataset
Includes cross-validation, threshold tuning, and metadata generation.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# Constants
URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
COLUMN_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
FEATURES_TO_CHECK = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def load_data(url=URL):
    """Load the Pima Indians Diabetes Dataset"""
    print("Loading dataset...")
    try:
        df = pd.read_csv(url, names=COLUMN_NAMES)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading from URL: {e}")
        return None

def preprocess_data(df):
    """Handle missing values and separate features/target"""
    df_copy = df.copy()
    
    # Handle missing values - 0s are often missing data in this dataset
    for feature in FEATURES_TO_CHECK:
        df_copy[feature] = df_copy[feature].replace(0, np.nan)
        median_value = df_copy[feature].median()
        df_copy[feature].fillna(median_value, inplace=True)
    
    X = df_copy.drop('Outcome', axis=1)
    y = df_copy['Outcome']
    return X, y

def find_best_threshold(y_true, y_probs):
    """Find the best threshold based on F1-score"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

def train_and_evaluate():
    """Main training pipeline"""
    df = load_data()
    if df is None:
        return
    
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predictions and Probs
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Threshold Tuning
    best_threshold = find_best_threshold(y_test, y_probs)
    print(f"\nOptimal Threshold (max F1): {best_threshold:.4f}")
    
    y_pred_tuned = (y_probs >= best_threshold).astype(int)
    y_pred_default = model.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred_tuned)
    f1 = f1_score(y_test, y_pred_tuned)
    
    print(f"\nTuned Model Accuracy: {accuracy:.4f}")
    print(f"Tuned Model F1-Score: {f1:.4f}")
    print("\nClassification Report (Tuned Threshold):")
    print(classification_report(y_test, y_pred_tuned, target_names=['Not Diabetic', 'Diabetic']))
    
    # Feature Importance
    importances = model.feature_importances_
    feature_importance = dict(zip(X.columns, importances.tolist()))
    
    # Metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "dataset_size": len(df),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "test_accuracy": float(accuracy),
        "test_f1": float(f1),
        "optimal_threshold": float(best_threshold),
        "feature_importance": feature_importance
    }
    
    # Determine outputs directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save artifacts
    print("\nSaving model, scaler, and metadata to models/ directory...")
    joblib.dump(model, os.path.join(models_dir, 'diabetes_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    with open(os.path.join(models_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Files saved: models/diabetes_model.pkl, models/scaler.pkl, models/metadata.json")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    train_and_evaluate()
