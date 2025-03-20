import tensorflow_decision_forests as tfdf
from config import NUM_TREES, MAX_DEPTH

def create_model():
    model = tfdf.keras.GradientBoostedTreesModel(
        num_trees=NUM_TREES,
        max_depth=MAX_DEPTH,
        verbose=1
    )
    return model 