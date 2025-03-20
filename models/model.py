from lightgbm import LGBMRegressor
from config.config import NUM_TREES, MAX_DEPTH

def create_model():
    model = LGBMRegressor(
        n_estimators=3000,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.01,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.01,
        random_state=42,
        force_col_wise=True,
        verbose=10,
        min_split_gain=0,
        boosting_type='gbdt',
        objective='regression',
        n_jobs=-1,
        metric='rmse',
        early_stopping_rounds=100
    )
    return model 