import joblib
import os
import sys
import lightgbm
import numpy as np
from time import time

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import load_and_preprocess_data, split_data
from models.model import create_model
from config.config import SAVED_MODELS_DIR, TARGET_FEATURE
from src.logger import setup_logger

# Setup logger
logger = setup_logger()

def train():
    try:
        logger.info("Starting training pipeline")
        start_time = time()
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        train_df, _ = load_and_preprocess_data()
        X_train, X_val, y_train, y_val = split_data(train_df)
        
        # Create and train model
        logger.info("Creating and training model")
        model = create_model()
        
        # Train with evaluation callback
        eval_result = {}
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=['train', 'valid'],
            eval_metric=['rmse', 'l2'],
            callbacks=[
                lightgbm.log_evaluation(period=10),  # Log every 10 iterations
                lightgbm.early_stopping(100),        # Early stopping
                lightgbm.record_evaluation(eval_result)  # Record metrics
            ]
        )
        
        # Calculate training time
        training_time = time() - start_time
        
        # Log final metrics
        best_iteration = model.best_iteration_
        best_score = model.best_score_
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best iteration: {best_iteration}")
        logger.info(f"Best validation RMSE: {best_score['valid']['rmse']:.6f}")
        logger.info(f"Best training RMSE: {best_score['train']['rmse']:.6f}")
        
        # Save model
        model_path = os.path.join(SAVED_MODELS_DIR, 'house_prices_model.joblib')
        joblib.dump(model, model_path)
        logger.success(f"Model successfully saved to {model_path}")
        
        # Save training history
        history_path = os.path.join(SAVED_MODELS_DIR, 'training_history.joblib')
        joblib.dump(eval_result, history_path)
        logger.info(f"Training history saved to {history_path}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    train()