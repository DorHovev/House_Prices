import tensorflow as tf
from House_Prices.src.data_preprocessing import load_and_preprocess_data, split_data
from models.model import create_model
from config.config import SAVED_MODELS_DIR, TARGET_FEATURE
from src.utils.logger import setup_logger # type: ignore
import os

# Setup logger
logger = setup_logger()

def train():
    try:
        logger.info("Starting training pipeline")
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        train_df, _ = load_and_preprocess_data()
        X_train, X_val, y_train, y_val = split_data(train_df)
        
        # Create TF datasets
        logger.info("Creating TensorFlow datasets")
        train_ds = tf.data.Dataset.from_tensor_slices((dict(X_train), y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((dict(X_val), y_val))
        
        # Create and train model
        logger.info("Creating and training model")
        model = create_model()
        model.fit(train_ds, validation_data=val_ds)
        
        # Save model
        model_path = os.path.join(SAVED_MODELS_DIR, 'house_prices_model')
        model.save(model_path)
        logger.success(f"Model successfully saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train() 