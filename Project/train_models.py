"""
Model Training Script for Cyber Attack Detection
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Create models directory
os.makedirs('models', exist_ok=True)

print("="*60)
print("Training Cyber Attack Detection Models")
print("="*60)

# Load dataset
df = pd.read_csv('realTrafficDataset.csv')
print(f"Dataset shape: {df.shape}")

# Features for prediction
features = [
    'hour', 'day_of_week', 'is_weekend',
    'url_length', 'url_num_params', 'url_num_special_chars',
    'url_path_depth', 'url_param_ratio', 'special_chars_ratio',
    'ua_length', 'ua_is_browser', 'ua_is_bot',
    'payload_special_chars',
    'payload_num_sql_keywords',
    'payload_num_xss_patterns',
    'source_ip_freq',
    'destination_ip_freq'
]

# Check available features
available_features = [f for f in features if f in df.columns]
print(f"Available features: {len(available_features)}")

X = df[available_features].copy()
y_binary = df['is_attack'].copy()
y_multi = df['attack_type_label'].copy()

# Preprocessing
print("\nPreprocessing data...")
for col in X.columns:
    if X[col].dtype in ['float64', 'int64']:
        X[col] = X[col].clip(lower=0)
        X[col] = X[col].fillna(0)

# Log transform skewed features
for col in ['source_ip_freq', 'destination_ip_freq']:
    if col in X.columns:
        X[col] = np.log1p(X[col].clip(lower=0))

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train_bin, y_test_bin = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_scaled, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

# Train Random Forest (Binary)
print("\nTraining Random Forest (Binary)...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train_bin)
print(f"Binary Accuracy: {rf_model.score(X_test, y_test_bin):.4f}")

# Train XGBoost (Multi-class)
print("\nTraining XGBoost (Multi-class)...")
label_encoder = LabelEncoder()
y_train_multi_encoded = label_encoder.fit_transform(y_train_multi)
y_test_multi_encoded = label_encoder.transform(y_test_multi)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_multi, y_train_multi_encoded)
print(f"Multi-class Accuracy: {xgb_model.score(X_test_multi, y_test_multi_encoded):.4f}")

# Save models and preprocessors
print("\nSaving models...")
joblib.dump(rf_model, 'models/binary_rf_model.pkl')
joblib.dump(xgb_model, 'models/multi_xgb_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
joblib.dump(available_features, 'models/feature_names.pkl')

print("\n✅ Training completed! Models saved in 'models/' directory.")