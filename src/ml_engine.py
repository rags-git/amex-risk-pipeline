import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def custom_metric_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.asarray(y_true)
    preds = np.asarray(y_pred)
    weight = 20.0 - labels * 19.0
    idx = np.argsort(preds)[::-1]
    labels = labels[idx]
    weight = weight[idx]
    
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_mask = cum_norm_weight <= 0.04
    d = labels[four_pct_mask].sum() / labels.sum()
    
    lorentz = (labels * weight).cumsum() / (labels * weight).sum()
    gini = sum(weight * (lorentz - cum_norm_weight))
    
    ideal_idx = np.argsort(labels)[::-1]
    ideal_labels = labels[ideal_idx]
    ideal_weight = weight[ideal_idx]
    ideal_cum_norm_weight = (ideal_weight / ideal_weight.sum()).cumsum()
    ideal_lorentz = (ideal_labels * ideal_weight).cumsum() / (ideal_labels * ideal_weight).sum()
    ideal_gini = sum(ideal_weight * (ideal_lorentz - ideal_cum_norm_weight))
    
    g = gini / ideal_gini
    return 0.5 * (g + d)

def lgb_custom_eval(preds, train_data):
    labels = train_data.get_label()
    score = custom_metric_numpy(labels, preds)
    return 'custom_metric', score, True

class CreditMLEngine:
    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits
        self.params = {
            'objective': 'binary',
            'metric': 'None',
            'boosting_type': 'gbdt',
            'n_jobs': -1,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'scale_pos_weight': 3.0,
            'verbose': -1
        }
        
    def train_cv(self, df: pd.DataFrame, target_col: str):
        X = df.drop(columns=[target_col, 'customer_ID'], errors='ignore')
        y = df[target_col].values
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        models = []
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training Fold {fold + 1}/{self.n_splits}...")
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]
            
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            model = lgb.train(
                params=self.params,
                train_set=dtrain,
                valid_sets=[dtrain, dval],
                num_boost_round=1500,
                callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)],
                feval=lgb_custom_eval
            )
            
            val_preds = model.predict(X_val, num_iteration=model.best_iteration)
            score = custom_metric_numpy(y_val, val_preds)
            logger.info(f"Fold {fold + 1} Validation Score: {score:.5f}")
            scores.append(score)
            models.append(model)
            
        logger.info(f"Mean OOF Score: {np.mean(scores):.5f}")
        joblib.dump(models[0], "models/amex_production_lgb.pkl")
        logger.info("Best production model artifact saved successfully.")

if __name__ == "__main__":
    pass
