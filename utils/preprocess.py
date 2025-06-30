import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_data(path):
    """Load data from CSV file."""
    df = pd.read_csv(path, low_memory=False)
    return df

def initial_inspection(df):
    """Print basic info: shape, head, info, missing summary."""
    print("Shape:", df.shape)
    display(df.head())
    print(df.info())
    print("Missing values per column:")
    print(df.isnull().sum())

def identify_columns(df, data_dict=None):
    """Identify numeric and categorical columns."""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_cols, categorical_cols

def build_preprocessing_pipeline(numeric_cols, categorical_cols, 
                                 impute_strategy='median', scaling=True):
    """Create ColumnTransformer for preprocessing."""
    # Numeric pipeline: impute and scale
    numeric_transformers = []
    if impute_strategy:
        numeric_transformers.append(('imputer', SimpleImputer(strategy=impute_strategy)))
    if scaling:
        numeric_transformers.append(('scaler', StandardScaler()))
    numeric_pipeline = Pipeline(numeric_transformers)

    # Categorical pipeline: impute (most frequent) and one-hot encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    return preprocessor

def save_preprocessor(preprocessor, path):
    """Save the fitted preprocessor using joblib."""
    import joblib
    joblib.dump(preprocessor, path)

def load_preprocessor(path):
    """Load a saved preprocessor."""
    import joblib
    return joblib.load(path)