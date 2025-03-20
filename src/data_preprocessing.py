import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import TRAIN_PATH, TEST_PATH, TARGET_FEATURE

def load_and_preprocess_data():
    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Handle missing values
    for df in [train_df, test_df]:
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column].fillna('missing', inplace=True)
            else:
                df[column].fillna(df[column].mean(), inplace=True)
    
    return train_df, test_df

def split_data(train_df, validation_size=0.2):
    X = train_df.drop(columns=[TARGET_FEATURE])
    y = train_df[TARGET_FEATURE]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_size, random_state=42
    )
    
    return X_train, X_val, y_train, y_val 