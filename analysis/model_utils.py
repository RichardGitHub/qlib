import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def save_model(model, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"[model_utils] Model saved to {filepath}")

def load_model(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file {filepath} not found.")
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"[model_utils] Model loaded from {filepath}")
    return model

def normalize_scores(
    pred_df: pd.DataFrame,
    method: str = "zscore",
    group_by_date: bool = True,
) -> pd.DataFrame:
    pred_df = pred_df.copy()
    if method not in {"zscore", "minmax", "robust"}:
        raise ValueError(f"Unsupported normalization method: {method}")
    scaler_cls = {
        "zscore": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }[method]

    if group_by_date and 'datetime' in pred_df.columns:
        pred_df['score_norm'] = pred_df.groupby('datetime')['score'].transform(
            lambda x: scaler_cls().fit_transform(x.values.reshape(-1, 1)).flatten()
        )
    else:
        scaler = scaler_cls()
        pred_df['score_norm'] = scaler.fit_transform(pred_df[['score']])

    return pred_df

def set_device_to_model(model, device="cpu"):
    """
    针对支持设备选择的模型，设置device。
    这里对TransformerModel等PyTorch模型有效。
    """
    # 递归给子模型设置device
    if hasattr(model, "models"):
        for sub_model, _ in model.models:
            set_device_to_model(sub_model, device)
    else:
        # 只对部分模型支持device参数的设置
        if hasattr(model, "device"):
            model.device = device
        elif hasattr(model, "set_device"):
            model.set_device(device)
    print(f"[model_utils] Set device={device} for model {type(model).__name__}")
