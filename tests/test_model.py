import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import model.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import load_data, preprocess_data, find_best_threshold

def test_preprocess_data():
    # Create fake dataframe mimicking Pima dataset
    data = {
        'Pregnancies': [1, 0, 5],
        'Glucose': [100, 0, 150],
        'BloodPressure': [70, 0, 80],
        'SkinThickness': [20, 0, 30],
        'Insulin': [80, 0, 120],
        'BMI': [25.0, 0.0, 30.0],
        'DiabetesPedigreeFunction': [0.5, 0.1, 0.8],
        'Age': [30, 25, 45],
        'Outcome': [0, 0, 1]
    }
    df = pd.DataFrame(data)
    
    X, y = preprocess_data(df)
    
    # 0 values should be replaced by median in X
    assert 0 not in X['Glucose'].values
    assert 0 not in X['BloodPressure'].values
    assert 0 not in X['SkinThickness'].values
    assert 0 not in X['Insulin'].values
    assert 0.0 not in X['BMI'].values
    
    # Shape should be correct
    assert X.shape == (3, 8)
    assert len(y) == 3

def test_find_best_threshold():
    # Create fake true labels and predicted probabilities
    y_true = np.array([0, 1, 0, 1, 1])
    y_probs = np.array([0.1, 0.9, 0.2, 0.8, 0.6])
    
    threshold = find_best_threshold(y_true, y_probs)
    # The optimal threshold should separate the 1s and 0s effectively
    assert 0.2 < threshold <= 0.6
