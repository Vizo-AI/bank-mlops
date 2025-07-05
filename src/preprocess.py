import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

RAW_PATH = Path('data/raw/default_of_credit_card_clients.xlsx')
PROC_DIR = Path('data/processed')
PROC_DIR.mkdir(exist_ok=True, parents=True)

def load_raw():
    return pd.read_excel(RAW_PATH, header=1).rename(columns={'default payment next month': 'default'})

def build_pipeline(df):
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    num_cols = [c for c in df.columns if c not in cat_cols + ['default', 'ID']]
    pre = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )
    return pre, num_cols, cat_cols

def main():
    df = load_raw()
    X = df.drop(['default', 'ID'], axis=1)
    y = df['default']
    pre, *_ = build_pipeline(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                       test_size=0.2, random_state=42, stratify=y)
    pre.fit(X_train)
    joblib.dump(pre, PROC_DIR/'prep.joblib')
    X_train_p = pre.transform(X_train)
    X_test_p  = pre.transform(X_test)
    # Save numpy arrays to keep memory small
    import numpy as np
    np.save(PROC_DIR/'X_train.npy', X_train_p)
    np.save(PROC_DIR/'X_test.npy',  X_test_p)
    np.save(PROC_DIR/'y_train.npy', y_train.to_numpy())
    np.save(PROC_DIR/'y_test.npy',  y_test.to_numpy())

if __name__ == "__main__":
    main()
