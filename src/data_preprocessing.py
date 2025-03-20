import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
import joblib

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import TRAIN_PATH, TEST_PATH, TARGET_FEATURE, SAVED_MODELS_DIR

def load_and_preprocess_data():
    """Load and preprocess both train and test data."""
    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print(f"Original train shape: {train_df.shape}")
    print(f"Original test shape: {test_df.shape}")
    
    # Log transform the target variable
    if TARGET_FEATURE in train_df.columns:
        train_df[TARGET_FEATURE] = np.log1p(train_df[TARGET_FEATURE])
    
    # Add target column to test_df with NaN values
    if TARGET_FEATURE not in test_df.columns:
        test_df[TARGET_FEATURE] = np.nan
    
    # Combine train and test for preprocessing
    combined_df = pd.concat([train_df, test_df], axis=0, sort=False)
    print(f"Combined shape before processing: {combined_df.shape}")
    
    # Handle missing values
    for column in combined_df.columns:
        if combined_df[column].dtype == 'object':
            combined_df[column].fillna('missing', inplace=True)
        else:
            # For numeric columns, fill with median from training data only
            train_median = train_df[column].median()
            combined_df[column].fillna(train_median, inplace=True)
    
    # Handle numeric features
    numeric_features = combined_df.select_dtypes(include=['int64', 'float64']).columns
    for column in numeric_features:
        if column != TARGET_FEATURE:
            skew = combined_df[column].skew()
            if skew > 0.75:
                combined_df[column] = np.log1p(combined_df[column] - combined_df[column].min() + 1)
    
    # Convert categorical variables to numeric
    categorical_columns = combined_df.select_dtypes(include=['object']).columns
    combined_df = pd.get_dummies(combined_df, columns=categorical_columns)
    
    # Clean column names
    combined_df.columns = [col.replace(' ', '_') for col in combined_df.columns]
    
    # Split back into train and test
    train_rows = len(train_df)
    train_df = combined_df.iloc[:train_rows].copy()
    test_df = combined_df.iloc[train_rows:].copy()
    
    print(f"Processed train shape: {train_df.shape}")
    print(f"Processed test shape: {test_df.shape}")
    
    # Drop target from test
    if TARGET_FEATURE in test_df.columns:
        test_df = test_df.drop(columns=[TARGET_FEATURE])
    
    # Save the column information for later use
    model_columns = test_df.columns  # Save test columns as they don't include target
    model_columns_path = os.path.join(SAVED_MODELS_DIR, 'model_columns.joblib')
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    joblib.dump(model_columns, model_columns_path)
    print(f"Saved {len(model_columns)} model columns to {model_columns_path}")
    
    # Verify no empty datasets
    if train_df.empty or test_df.empty:
        raise ValueError("Either train or test dataset is empty after preprocessing")
    
    return train_df, test_df

def split_data(train_df, validation_size=0.2):
    X = train_df.drop(columns=[TARGET_FEATURE])
    y = train_df[TARGET_FEATURE]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_size, random_state=42
    )
    
    return X_train, X_val, y_train, y_val

def load_and_preprocess_test_data():
    """Load and preprocess only the test data."""
    # Load test data
    test_df = pd.read_csv(TEST_PATH)
    print(f"Original test shape: {test_df.shape}")
    
    # Load the column information
    model_columns_path = os.path.join(SAVED_MODELS_DIR, 'model_columns.joblib')
    if not os.path.exists(model_columns_path):
        print("Model columns file not found. Processing full dataset...")
        train_processed, test_processed = load_and_preprocess_data()
        return test_processed
    
    model_columns = joblib.load(model_columns_path)
    print(f"Loaded {len(model_columns)} model columns")
    
    # Process test data using the same steps as in training
    # Handle missing values
    for column in test_df.columns:
        if test_df[column].dtype == 'object':
            test_df[column].fillna('missing', inplace=True)
        else:
            test_df[column].fillna(test_df[column].median(), inplace=True)
    
    # Handle numeric features
    numeric_features = test_df.select_dtypes(include=['int64', 'float64']).columns
    for column in numeric_features:
        skew = test_df[column].skew()
        if skew > 0.75:
            test_df[column] = np.log1p(test_df[column] - test_df[column].min() + 1)
    
    # Convert categorical variables to numeric
    categorical_columns = test_df.select_dtypes(include=['object']).columns
    test_df = pd.get_dummies(test_df, columns=categorical_columns)
    
    # Clean column names
    test_df.columns = [col.replace(' ', '_') for col in test_df.columns]
    
    # Ensure all necessary columns exist
    for column in model_columns:
        if column not in test_df.columns:
            test_df[column] = 0
    
    # Select only the required columns in the correct order
    test_df = test_df[model_columns]
    
    print(f"Processed test shape: {test_df.shape}")
    
    return test_df 