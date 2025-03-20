import joblib
import os
import sys
import numpy as np
from time import time
import pandas as pd

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import load_and_preprocess_data, load_and_preprocess_test_data
from config.config import SAVED_MODELS_DIR
from src.logger import setup_logger

# Setup logger
logger = setup_logger()

def predict():
    try:
        logger.info("Starting prediction pipeline")
        
        # Load and preprocess test data
        logger.info("Loading and preprocessing test data")
        test_df = load_and_preprocess_test_data()
        
        # Validate test data
        if test_df is None:
            raise ValueError("Test data failed to load")
        if test_df.empty:
            raise ValueError("Test data is empty after preprocessing")
        if test_df.isnull().any().any():
            raise ValueError("Test data contains null values after preprocessing")
        
        logger.info(f"Test data shape: {test_df.shape}")
        
        # Load model
        model_path = os.path.join(SAVED_MODELS_DIR, 'house_prices_model.joblib')
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Make predictions
        logger.info("Making predictions")
        predictions = model.predict(test_df)
        
        # Convert predictions back from log scale
        predictions = np.expm1(predictions)
        
        logger.success("Predictions completed successfully")
        
        # Create a results DataFrame
        results = pd.DataFrame({
            'Id': range(1461, 1461 + len(predictions)),  # Test data IDs
            'SalePrice': predictions
        })
        
        # Save predictions to CSV
        output_path = os.path.join(SAVED_MODELS_DIR, 'predictions.csv')
        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Display first few predictions
        logger.info("\nFirst 5 predictions:")
        logger.info(results.head())
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    predictions = predict()
    logger.info(f"Predictions shape: {predictions.shape}")