import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

# Support both datasets - new Kaggle real dataset is primary
def get_feature_lists():
    real_path = "artifacts/customers.csv"
    if os.path.exists(real_path):
        try:
            df = pd.read_csv(real_path, nrows=5)
            if 'customer_segment' in df.columns:
                numeric = ['age','annual_income','months_active','avg_monthly_spend','purchase_frequency','avg_order_value','discount_usage_rate','return_rate','browsing_time_minutes','support_interactions']
                categorical = ['payment_method','region']
                return numeric, categorical, 'customer_segment'
        except:
            pass
    numeric = ['Age','AnnualIncome','SpendingScore','TenureMonths','NumPurchases','AvgOrderValue','RecencyDays','WebVisitsPerMonth','SupportTickets']
    categorical = ['Gender','Region','MembershipTier']
    return numeric, categorical, 'CustomerCategory'

NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN = get_feature_lists()

FEATURE_STORE = {
    "description": "Centralized feature definitions with versioning - real Kaggle dataset",
    "version": "v3.0-kaggle",
    "dataset_source": "Kaggle - Online Retail Customer Segmentation Dataset (rohit8527kmr7518) - 50,000 customers",
    "dataset_file": "retail_customer_segmentation.csv",
    "rows": 50000,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "target": TARGET_COLUMN,
    "classes": ["Occasional","Regular","Loyal","High_Value"],
    "class_distribution": {"Occasional":22109,"Regular":13455,"Loyal":9163,"High_Value":5273},
    "missing_values": {"annual_income":3075,"avg_monthly_spend":2520,"purchase_frequency":1979,"discount_usage_rate":2549,"return_rate":2487,"browsing_time_minutes":3934,"support_interactions":1988},
    "transformations": {
        "numeric": ["median_imputation", "standard_scaling"],
        "categorical": ["most_frequent_imputation", "one_hot_encoding"]
    },
    "created_at": "2025-09-21",
    "training_serving_skew_prevention": [
        "Same imputer statistics used in training and serving",
        "Same scaler mean/std reused",
        "Same one-hot categories reused",
        "Feature order enforced via ColumnTransformer",
        "Input validation schema checked at inference",
        "customer_id ignored (not a feature)"
    ]
}

def get_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
    return preprocessor

def build_feature_names(preprocessor):
    num_features = NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    return list(num_features) + list(cat_features)

def validate_input(df):
    errors = []
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    return errors

def save_preprocessor(preprocessor, path="artifacts/preprocessor.pkl"):
    joblib.dump(preprocessor, path)
    joblib.dump(FEATURE_STORE, path.replace('preprocessor.pkl','feature_store.pkl'))

def load_preprocessor(path="artifacts/preprocessor.pkl"):
    return joblib.load(path)
