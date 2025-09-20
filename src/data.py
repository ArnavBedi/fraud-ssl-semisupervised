import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

def load_data(path, target='Class', id_col=None):
    df = pd.read_csv(path)
    assert target in df.columns, f"Target column '{target}' not found. Got: {df.columns.tolist()}"
    if id_col and id_col in df.columns:
        df = df.sort_values(id_col)
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y, df

def make_splits(X, y, test_size=0.2, val_size=0.2, seed=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_ratio = val_size / (1 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=seed
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def make_numeric_pipeline(strategy="standard"):
    scaler = StandardScaler() if strategy == "standard" else RobustScaler()
    return Pipeline(steps=[("scaler", scaler)])
