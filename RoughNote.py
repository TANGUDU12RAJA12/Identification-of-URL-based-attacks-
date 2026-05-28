# import pandas as pd
# df = pd.read_csv('cyber_attack_dataset_final.csv')




# # ==============================
# # BINARY MODEL FEATURE SELECTION
# # ==============================

# binary_features = [

#     # Time-based (processed)
#     'hour', 'day_of_week', 'is_weekend',

#     # URL Features
#     'url_length', 'url_num_params', 'url_num_special_chars',
#     'url_has_sql_keywords', 'url_has_xss_patterns', 'url_has_cmd_injection',
#     'url_has_traversal', 'url_has_ssrf_patterns', 'url_has_typosquatting',
#     'url_contains_ip', 'url_has_credential_patterns', 'url_path_depth',
#     'url_has_sensitive_file',

#     # Payload Features
#     'payload_length', 'request_body_length', 'payload_special_chars',
#     'payload_num_sql_keywords', 'payload_num_xss_patterns',
#     'payload_has_encoding', 'payload_is_empty', 'payload_contains_hex',
#     'payload_length_safe',

#     # Response Features
#     'is_error_response', 'is_success_response', 'is_redirect_response',
#     'response_size_ratio', 'response_payload_ratio',

#     # IP Behavior Features
#     'source_ip_freq', 'source_ip_attack_ratio', 'is_known_attacker_ip',
#     'destination_ip_freq', 'destination_ip_attack_count',
#     'requests_per_ip_per_minute', 'source_ip_error_ratio',

#     # User-Agent Features
#     'is_attack_tool_ua', 'ua_is_browser', 'ua_is_bot', 'ua_length',

#     # Attack Pattern Features (Allowed for binary)
#     'attack_pattern_sql', 'attack_pattern_xss', 'attack_pattern_cmd',
#     'attack_pattern_traversal', 'attack_pattern_ssrf',
#     'attack_pattern_typosquatting', 'attack_pattern_credential',

#     # Ratio Features
#     'special_chars_ratio', 'url_param_ratio',

#     # Protocol & Method
#     'is_https', 'is_post_method', 'is_get_method'
# ]

# # Final dataset for model
# X = df[binary_features]
# y = df['is_attack']

# # Check
# print("Feature Shape:", X.shape)
# print("Target Shape:", y.shape)


# # ==============================
# # DATA VALIDATION BEFORE MODEL
# # ==============================

# print("="*60)
# print("1. SHAPE CHECK")
# print("="*60)
# print("X Shape:", X.shape)
# print("y Shape:", y.shape)


# # ==============================
# # 2. MISSING VALUES
# # ==============================
# print("\n" + "="*60)
# print("2. MISSING VALUES")
# print("="*60)

# missing = X.isnull().sum()
# print(missing[missing > 0])

# print("\nTotal Missing Values:", X.isnull().sum().sum())


# # ==============================
# # 3. DATA TYPES CHECK
# # ==============================
# print("\n" + "="*60)
# print("3. DATA TYPES")
# print("="*60)

# print(X.dtypes.value_counts())
# print("\nNon-numeric columns:")
# print(X.select_dtypes(include='object').columns)


# # ==============================
# # 4. UNIQUE VALUES (for categorical-like)
# # ==============================
# print("\n" + "="*60)
# print("4. LOW CARDINALITY CHECK")
# print("="*60)

# for col in X.columns:
#     unique_vals = X[col].nunique()
#     if unique_vals < 10:
#         print(f"{col}: {unique_vals} unique values")


# # ==============================
# # 5. VALUE RANGE CHECK
# # ==============================
# print("\n" + "="*60)
# print("5. STATISTICAL SUMMARY")
# print("="*60)

# print(X.describe())


# # ==============================
# # 6. TARGET VARIABLE CHECK
# # ==============================
# print("\n" + "="*60)
# print("6. TARGET DISTRIBUTION")
# print("="*60)

# print(y.value_counts())
# print("\nTarget Unique Values:", y.unique())


# # ==============================
# # 7. CHECK FOR INFINITE VALUES
# # ==============================
# print("\n" + "="*60)
# print("7. INFINITE VALUES CHECK")
# print("="*60)

# import numpy as np
# print("Infinite values:", np.isinf(X).sum().sum())


# # ==============================
# # 8. FINAL VERDICT
# # ==============================
# print("\n" + "="*60)
# print("FINAL CHECKLIST")
# print("="*60)

# if X.isnull().sum().sum() == 0 and len(X.select_dtypes(include='object').columns) == 0:
#     print("✅ Data is clean and ready for model training")
# else:
#     print("⚠️ Data needs preprocessing before training")


## Above code is for data validation before model training. It checks for shape, missing values, data types, unique values, value ranges, target distribution, and infinite values. The final verdict indicates if the data is ready for modeling or needs preprocessing.