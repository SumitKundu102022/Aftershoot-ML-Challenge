import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump, load


LABEL_COLS = ["Temperature", "Tint"]
ID_COL = "id_global"


def select_feature_columns(df, label_cols=LABEL_COLS, id_col=ID_COL):
    """Return lists of numeric and categorical feature columns, excluding id + labels."""
    cols_to_exclude = list(label_cols) + [id_col]
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]

    numeric_cols = []
    categorical_cols = []

    for col in feature_cols:
        # Try to detect numeric columns robustly
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            # Try coercing to numeric; if many values convert, treat as numeric
            coerced = pd.to_numeric(df[col], errors="coerce")
            non_na_ratio = coerced.notna().mean()
            if non_na_ratio > 0.9:  # mostly numeric
                numeric_cols.append(col)
                df[col] = coerced
            else:
                categorical_cols.append(col)

    return feature_cols, numeric_cols, categorical_cols


def fit_preprocessors(train_df):
    """
    Fit StandardScaler for numeric features and OneHotEncoder for categorical.
    Returns a dict with everything needed to transform data later.
    """
    feature_cols, numeric_cols, categorical_cols = select_feature_columns(train_df)

    # Numeric
    X_num = None
    num_scaler = None
    if numeric_cols:
        X_num = train_df[numeric_cols].copy()
        X_num = X_num.fillna(X_num.median())
        num_scaler = StandardScaler()
        num_scaler.fit(X_num)

    # Categorical
    X_cat = None
    cat_encoder = None
    if categorical_cols:
        X_cat = train_df[categorical_cols].copy().fillna("Unknown").astype(str)
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_encoder.fit(X_cat)

    preprocessors = {
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_scaler": num_scaler,
        "cat_encoder": cat_encoder,
    }

    return preprocessors


def transform_metadata(df, preprocessors):
    numeric_cols = preprocessors["numeric_cols"]
    categorical_cols = preprocessors["categorical_cols"]
    num_scaler = preprocessors["numeric_scaler"]
    cat_encoder = preprocessors["cat_encoder"]

    pieces = []

    # Numeric columns that exist in df
    valid_numeric = [c for c in numeric_cols if c in df.columns]
    X_num_all = np.zeros((len(df), len(numeric_cols)))
    
    for i, col in enumerate(numeric_cols):
        if col in df.columns:
            col_data = df[col].fillna(df[col].median())
            X_num_all[:, i] = col_data
        else:
            X_num_all[:, i] = 0
            
    X_num_scaled = num_scaler.transform(X_num_all)
    pieces.append(X_num_scaled)
    
    # Categorical columns that exist in df
    valid_categorical = [c for c in categorical_cols if c in df.columns]
    if valid_categorical:
        X_cat = df[valid_categorical].copy().fillna("Unknown").astype(str)
        X_cat_enc = cat_encoder.transform(X_cat)
        pieces.append(X_cat_enc)

    if len(pieces) == 0:
        raise ValueError("No valid metadata columns found in dataframe.")

    X_meta = np.concatenate(pieces, axis=1)
    return X_meta



def save_preprocessors(preprocessors, path):
    dump(preprocessors, path)


def load_preprocessors(path):
    return load(path)
