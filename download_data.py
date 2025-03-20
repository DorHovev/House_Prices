import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger
from config.config import DATA_DIR

def download_dataset():
    try:
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        logger.info("Initializing Kaggle API")
        api = KaggleApi()
        api.authenticate()
        
        logger.info("Downloading dataset files...")
        api.competition_download_files(
            'house-prices-advanced-regression-techniques',
            path=DATA_DIR
        )
        
        # Unzip the downloaded file
        zip_file = os.path.join(DATA_DIR, 'house-prices-advanced-regression-techniques.zip')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        # Remove zip file
        os.remove(zip_file)
        
        logger.success("Dataset downloaded and extracted successfully!")

    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

if __name__ == "__main__":
    download_dataset()