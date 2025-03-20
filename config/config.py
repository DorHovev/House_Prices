import os


# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.getenv('DATA_PATH', os.path.join(BASE_DIR, 'data'))
SAVED_MODELS_DIR = os.getenv('MODELS_PATH', os.path.join(BASE_DIR, 'saved_models'))
LOGS_DIR = os.getenv('LOGS_PATH', os.path.join(BASE_DIR, 'logs'))

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Data paths
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

# Model settings
TARGET_FEATURE = 'SalePrice'

# Training parameters
NUM_TREES = 300
MAX_DEPTH = 8 

