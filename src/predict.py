import tensorflow as tf
from House_Prices.src.data_preprocessing import load_and_preprocess_data
from House_Prices.config.config import SAVED_MODELS_DIR
from House_Prices.src.logger import setup_logger
import os

# Setup logger
logger = setup_logger()

def predict():
    try:
        logger.info("Starting prediction pipeline")
        
        # Load test data
        logger.info("Loading test data")
        _, test_df = load_and_preprocess_data()
        
        # Load model
        model_path = os.path.join(SAVED_MODELS_DIR, 'house_prices_model')
        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        # Make predictions
        logger.info("Making predictions")
        predictions = model.predict(dict(test_df))
        logger.success("Predictions completed successfully")
        return predictions
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    predictions = predict()
    logger.info(f"Predictions shape: {predictions.shape}")