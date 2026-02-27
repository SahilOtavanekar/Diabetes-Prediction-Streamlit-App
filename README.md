# Diabetes Prediction App

A Streamlit-based web application for predicting diabetes risk using machine learning. This app uses a RandomForestClassifier trained on the Pima Indians Diabetes Dataset.

## Features

- Interactive web interface built with Streamlit
- Real-time diabetes prediction based on health metrics
- Probability scores for both outcomes
- Clean and user-friendly design

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

**Note for Python 3.14 users:** 
- If you encounter pyarrow build errors, the app includes fallback methods and will work without pyarrow
- Streamlit will automatically install compatible dependencies, or the app will use alternative display methods
- The app is configured to handle pyarrow compatibility issues gracefully

## Usage

### Step 1: Train the Model

First, train the machine learning model and save it:

```bash
python model.py
```

This script will:
- Load the Pima Indians Diabetes Dataset
- Preprocess the data (handle missing values, standardize features)
- Train a RandomForestClassifier
- Save the model as `diabetes_model.pkl`
- Save the scaler as `scaler.pkl`
- Display model accuracy and classification report

### Step 2: Run the Streamlit App

After training the model, launch the Streamlit application:

**Option 1 (Recommended):**
```bash
python -m streamlit run app.py
```

**Option 2 (Alternative):**
```bash
python run_app.py
```

**Note:** If you get a "streamlit command not recognized" error, it means the Python Scripts folder is not in your PATH. Use Option 1 or Option 2 above, which work without PATH configuration.

The app will open in your default web browser. You can:
- Enter health metrics in the sidebar
- Click "Predict" to get diabetes prediction
- View probability scores and visualization

## Project Structure

```
.
├── app.py              # Streamlit web application
├── model.py            # Model training script
├── requirements.txt    # Python dependencies
├── diabetes_model.pkl  # Trained model (generated after running model.py)
├── scaler.pkl         # Feature scaler (generated after running model.py)
└── README.md          # This file
```

## Input Features

The model uses the following health metrics:
1. **Pregnancies** - Number of times pregnant
2. **Glucose** - Plasma glucose concentration (mg/dL)
3. **Blood Pressure** - Diastolic blood pressure (mm Hg)
4. **Skin Thickness** - Triceps skin fold thickness (mm)
5. **Insulin** - 2-Hour serum insulin (mu U/ml)
6. **BMI** - Body Mass Index (kg/m²)
7. **Diabetes Pedigree Function** - Family history score
8. **Age** - Age in years

## Model Details

- **Algorithm**: RandomForestClassifier
- **Training/Test Split**: 80/20
- **Preprocessing**: StandardScaler for feature normalization
- **Missing Value Handling**: Median imputation

## Important Notes

⚠️ **Medical Disclaimer**: This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## License

This project is provided as-is for educational purposes.

