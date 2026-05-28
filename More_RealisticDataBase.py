import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Read the original dataset
df = pd.read_csv('cyber_attack_dataset_final.csv')

# Display original dataset info
print("Original Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Attack distribution:\n{df['attack_type_label'].value_counts()}")
print(f"Normal samples: {len(df[df['attack_type_label'] == 'normal'])}")
print(f"Attack samples: {len(df[df['attack_type_label'] != 'normal'])}")

# ============================================
# 1. Introduce Class Imbalance (more normal traffic, fewer attacks)
# ============================================
print("\n1. Introducing class imbalance...")

# Keep all attack samples
attack_df = df[df['attack_type_label'] != 'normal']

# Increase normal samples (simulate more benign traffic)
normal_df = df[df['attack_type_label'] == 'normal']
# Add synthetic normal traffic with variations
additional_normal = []
for _ in range(3000):  # Add 3000 more normal samples
    if len(normal_df) > 0:
        sample = normal_df.sample(1).iloc[0].copy()
        
        # Modify timestamp to create realistic distribution
        if isinstance(sample['timestamp'], str):
            base_time = datetime.strptime(sample['timestamp'], '%Y-%m-%d %H:%M:%S')
            # Add random time variation within 24 hours
            time_delta = timedelta(hours=np.random.randint(-12, 12), 
                                   minutes=np.random.randint(-59, 59))
            sample['timestamp'] = (base_time + time_delta).strftime('%Y-%m-%d %H:%M:%S')
        
        # Modify user agent to increase diversity
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
            'Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15',
            'python-requests/2.28.0',
            'curl/7.81.0',
            'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
            'Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0'
        ]
        sample['User-Agent'] = np.random.choice(user_agents)
        sample['user_agent'] = sample['User-Agent']
        
        # Ensure it's classified as normal
        sample['attack_type_label'] = 'normal'
        sample['is_attack'] = 0
        
        # Clear any attack-specific features
        if 'payload_raw' in sample:
            sample['payload_raw'] = ''
        if 'payload_length' in sample:
            sample['payload_length'] = 0
        if 'payload_special_chars' in sample:
            sample['payload_special_chars'] = 0
            
        additional_normal.append(sample)

normal_df_augmented = pd.concat([normal_df, pd.DataFrame(additional_normal)], ignore_index=True)

# Create imbalanced dataset: 70% normal, 30% attacks (approx)
normal_count = len(normal_df_augmented)
attack_count = len(attack_df)
total_samples = normal_count + attack_count

print(f"After augmentation - Normal: {normal_count}, Attack: {attack_count}")
print(f"Imbalance ratio: {normal_count/attack_count:.2f}:1")

df_imbalanced = pd.concat([normal_df_augmented, attack_df], ignore_index=True)

# ============================================
# 2. Add Controlled Label Noise
# ============================================
print("\n2. Adding label noise (incorrect labels)...")

# Create a copy for label noise
df_with_noise = df_imbalanced.copy()

# Introduce label noise in 3-5% of samples
noise_ratio = 0.04  # 4% label noise
noise_indices = np.random.choice(df_with_noise.index, 
                                  size=int(len(df_with_noise) * noise_ratio), 
                                  replace=False)

# Get unique attack types (excluding 'normal')
attack_types = [t for t in df_with_noise['attack_type_label'].unique() if t != 'normal']

for idx in noise_indices:
    current_label = df_with_noise.loc[idx, 'attack_type_label']
    
    if current_label == 'normal':
        # Flip normal to random attack type
        new_label = np.random.choice(attack_types)
        df_with_noise.loc[idx, 'attack_type_label'] = new_label
        df_with_noise.loc[idx, 'is_attack'] = 1
        
        # Add some attack-like features to make it plausible
        if new_label == 'sql_injection':
            df_with_noise.loc[idx, 'url_has_sql_keywords'] = np.random.choice([0, 1], p=[0.3, 0.7])
            df_with_noise.loc[idx, 'url_num_special_chars'] += np.random.randint(3, 8)
        elif new_label == 'xss':
            df_with_noise.loc[idx, 'url_has_xss_patterns'] = np.random.choice([0, 1], p=[0.4, 0.6])
        elif new_label == 'command_injection':
            df_with_noise.loc[idx, 'url_has_cmd_injection'] = np.random.choice([0, 1], p=[0.5, 0.5])
            
    else:
        # Flip attack to normal or different attack type
        if np.random.random() < 0.5:
            # Flip to normal
            df_with_noise.loc[idx, 'attack_type_label'] = 'normal'
            df_with_noise.loc[idx, 'is_attack'] = 0
            # Clear attack indicators
            attack_cols = ['url_has_sql_keywords', 'url_has_xss_patterns', 
                          'url_has_cmd_injection', 'url_has_traversal',
                          'url_has_ssrf_patterns', 'url_has_typosquatting',
                          'url_has_credential_patterns']
            for col in attack_cols:
                if col in df_with_noise.columns:
                    df_with_noise.loc[idx, col] = 0
        else:
            # Flip to different attack type
            other_attacks = [t for t in attack_types if t != current_label]
            if other_attacks:
                new_label = np.random.choice(other_attacks)
                df_with_noise.loc[idx, 'attack_type_label'] = new_label

print(f"Added {len(noise_indices)} samples with label noise ({noise_ratio*100:.1f}%)")

# ============================================
# 3. Inject Feature Noise
# ============================================
print("\n3. Injecting feature noise...")

df_with_noise = df_with_noise.copy()

# Add Gaussian noise to numeric columns
numeric_cols = df_with_noise.select_dtypes(include=[np.number]).columns.tolist()
# Exclude binary indicators from heavy noise
binary_cols = [col for col in numeric_cols if df_with_noise[col].nunique() <= 2]
continuous_cols = [col for col in numeric_cols if col not in binary_cols]

for col in continuous_cols:
    if col in df_with_noise.columns:
        # Add noise scaled to 5-15% of standard deviation
        noise_std = df_with_noise[col].std() * np.random.uniform(0.05, 0.15)
        noise = np.random.normal(0, noise_std, len(df_with_noise))
        df_with_noise[col] = df_with_noise[col] + noise
        # Clip to non-negative where appropriate
        if col in ['payload_length', 'request_body_length', 'url_length']:
            df_with_noise[col] = df_with_noise[col].clip(lower=0)

# ============================================
# 4. Introduce Missing Values
# ============================================
print("\n4. Introducing missing values...")

# Introduce missing values in 2-5% of samples for selected columns
missing_cols = ['request_headers', 'response_headers', 'request_body', 'payload_raw', 
                'parsed_request_headers', 'parsed_response_headers']

missing_rate = 0.03  # 3% missing values

for col in missing_cols:
    if col in df_with_noise.columns:
        missing_indices = np.random.choice(df_with_noise.index, 
                                           size=int(len(df_with_noise) * missing_rate),
                                           replace=False)
        df_with_noise.loc[missing_indices, col] = np.nan

# ============================================
# 5. Create Overlap Between Normal and Attack Samples
# ============================================
print("\n5. Creating overlap between normal and attack samples...")

# Select some normal samples that will get attack-like features
overlap_ratio = 0.1  # 10% of normal samples will have attack-like features
normal_indices = df_with_noise[df_with_noise['attack_type_label'] == 'normal'].index
overlap_indices = np.random.choice(normal_indices, 
                                   size=int(len(normal_indices) * overlap_ratio),
                                   replace=False)

for idx in overlap_indices:
    # Add some suspicious features but keep label as normal
    if np.random.random() < 0.5:
        # Add SQL-like patterns
        df_with_noise.loc[idx, 'url_has_sql_keywords'] = np.random.choice([0, 1], p=[0.6, 0.4])
        df_with_noise.loc[idx, 'url_num_special_chars'] += np.random.randint(1, 3)
        # But keep overall attack flag as 0
        df_with_noise.loc[idx, 'is_attack'] = 0
    else:
        # Add XSS-like patterns
        df_with_noise.loc[idx, 'url_has_xss_patterns'] = np.random.choice([0, 1], p=[0.7, 0.3])
        df_with_noise.loc[idx, 'url_num_special_chars'] += np.random.randint(1, 3)

# ============================================
# 6. Simulate Time-based Patterns (Burst Traffic)
# ============================================
print("\n6. Simulating time-based patterns...")

# Sort by timestamp
df_with_noise['timestamp'] = pd.to_datetime(df_with_noise['timestamp'])
df_with_noise = df_with_noise.sort_values('timestamp').reset_index(drop=True)

# Create bursts of traffic from specific IPs
burst_ips = df_with_noise['source_ip'].value_counts().head(10).index.tolist()

for ip in burst_ips[:5]:  # Create bursts for top 5 IPs
    ip_indices = df_with_noise[df_with_noise['source_ip'] == ip].index
    if len(ip_indices) > 10:
        # Take a contiguous block of requests (burst)
        start_idx = np.random.randint(0, len(ip_indices) - 20)
        burst_indices = ip_indices[start_idx:start_idx + np.random.randint(10, 25)]
        
        # Modify timestamps to create close temporal proximity
        base_time = df_with_noise.loc[burst_indices[0], 'timestamp']
        for i, idx in enumerate(burst_indices):
            df_with_noise.loc[idx, 'timestamp'] = base_time + timedelta(seconds=i * np.random.randint(1, 5))

# ============================================
# 7. Add Feature Correlations
# ============================================
print("\n7. Adding feature correlations...")

# Correlate payload_length with url_num_special_chars
correlation_factor = 0.6
attack_mask = df_with_noise['is_attack'] == 1
attack_indices = df_with_noise[attack_mask].index

for idx in attack_indices:
    if 'payload_length' in df_with_noise.columns and 'url_num_special_chars' in df_with_noise.columns:
        # For attacks, payload length correlates with special characters
        current_special = df_with_noise.loc[idx, 'url_num_special_chars']
        if current_special > 0:
            # Payload length increases with special characters
            payload_boost = current_special * np.random.uniform(5, 15)
            df_with_noise.loc[idx, 'payload_length'] += payload_boost

# Correlate payload_length with response_size (for successful attacks)
success_indices = df_with_noise[(df_with_noise['response_code'] >= 200) & 
                                (df_with_noise['response_code'] < 300)].index

for idx in success_indices:
    if 'payload_length' in df_with_noise.columns and 'response_size' in df_with_noise.columns:
        if df_with_noise.loc[idx, 'payload_length'] > 0:
            # Longer payloads sometimes lead to larger responses
            response_boost = df_with_noise.loc[idx, 'payload_length'] * np.random.uniform(0.5, 2)
            df_with_noise.loc[idx, 'response_size'] += response_boost

# ============================================
# 8. Generate Borderline/Ambiguous Attack Samples
# ============================================
print("\n8. Generating borderline/ambiguous attack samples...")

# Take some normal samples and make them borderline
normal_samples = df_with_noise[df_with_noise['attack_type_label'] == 'normal'].copy()
borderline_count = int(len(normal_samples) * 0.05)  # 5% of normal become borderline

if borderline_count > 0:
    borderline_indices = np.random.choice(normal_samples.index, size=borderline_count, replace=False)
    
    for idx in borderline_indices:
        # Add some ambiguous features
        if np.random.random() < 0.7:
            # Add weak attack indicators
            df_with_noise.loc[idx, 'url_num_special_chars'] += np.random.randint(1, 4)
            df_with_noise.loc[idx, 'url_length'] += np.random.randint(5, 20)
            
            # Add one mild attack pattern
            pattern_type = np.random.choice(['sql', 'xss', 'cmd'])
            if pattern_type == 'sql':
                df_with_noise.loc[idx, 'url_has_sql_keywords'] = 1
                df_with_noise.loc[idx, 'url_has_sql_keywords'] = 1
            elif pattern_type == 'xss':
                df_with_noise.loc[idx, 'url_has_xss_patterns'] = 1
            else:
                df_with_noise.loc[idx, 'url_has_cmd_injection'] = 1
            
            # Keep label as normal for these ambiguous samples
            df_with_noise.loc[idx, 'is_attack'] = 0

# ============================================
# 9. Increase User-Agent Diversity
# ============================================
print("\n9. Increasing user-agent diversity...")

user_agents = [
    # Browsers
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
    
    # Bots
    'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
    'Mozilla/5.0 (compatible; Bingbot/2.0; +http://www.bing.com/bingbot.htm)',
    'Mozilla/5.0 (compatible; YandexBot/3.0; +http://yandex.com/bots)',
    
    # Scanning/Attack tools
    'BurpSuite/2023.12',
    'SQLmap/1.8.4',
    'XSStrike/3.1.5',
    'Nikto/2.5.0',
    'Nmap Scripting Engine',
    'curl/7.81.0',
    'python-requests/2.28.0',
    'wget/1.21.2',
    
    # Unknown/Custom
    'Mozilla/5.0 (Unknown; Unknown) Unknown',
    'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
]

# Randomly assign user agents with weighted probabilities
weights = [0.15] * 7 + [0.05] * 4 + [0.03] * 6 + [0.02] * 4  # Adjust weights for realistic distribution

df_with_noise['User-Agent'] = np.random.choice(user_agents, size=len(df_with_noise), p=weights/np.sum(weights))
df_with_noise['user_agent'] = df_with_noise['User-Agent']

# ============================================
# 10. Ensure Features Not Perfectly Aligned with Labels
# ============================================
print("\n10. Adding noise to feature-label relationships...")

# For a subset of attacks, reduce or remove attack indicators
attack_indices = df_with_noise[df_with_noise['is_attack'] == 1].index
obfuscate_ratio = 0.05  # 5% of attacks have obfuscated features

obfuscate_indices = np.random.choice(attack_indices, size=int(len(attack_indices) * obfuscate_ratio), replace=False)

for idx in obfuscate_indices:
    # Remove or reduce attack-specific features
    attack_cols = ['url_has_sql_keywords', 'url_has_xss_patterns', 'url_has_cmd_injection',
                   'url_has_traversal', 'url_has_ssrf_patterns', 'url_has_typosquatting']
    
    for col in attack_cols:
        if col in df_with_noise.columns:
            df_with_noise.loc[idx, col] = np.random.choice([0, 1], p=[0.7, 0.3])
    
    # Reduce payload special characters
    if 'payload_special_chars' in df_with_noise.columns:
        df_with_noise.loc[idx, 'payload_special_chars'] = max(0, df_with_noise.loc[idx, 'payload_special_chars'] - np.random.randint(2, 5))
    
    # Keep attack label to maintain label noise
    df_with_noise.loc[idx, 'is_attack'] = 1

# ============================================
# Post-processing and Final Checks
# ============================================
print("\nPerforming final post-processing...")

# Convert timestamp back to string
df_with_noise['timestamp'] = df_with_noise['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Recalculate derived features that might have been affected
if 'payload_length_safe' in df_with_noise.columns:
    df_with_noise['payload_length_safe'] = df_with_noise['payload_length'].apply(lambda x: 0 if x == 0 else 1)

if 'special_chars_ratio' in df_with_noise.columns and 'url_length' in df_with_noise.columns:
    df_with_noise['special_chars_ratio'] = df_with_noise['url_num_special_chars'] / df_with_noise['url_length'].replace(0, 1)

if 'response_payload_ratio' in df_with_noise.columns:
    df_with_noise['response_payload_ratio'] = df_with_noise['response_size'] / df_with_noise['payload_length'].replace(0, 1)

# Fill any remaining NaN values with appropriate defaults
for col in df_with_noise.columns:
    if df_with_noise[col].dtype in ['float64', 'int64']:
        df_with_noise[col] = df_with_noise[col].fillna(0)
    else:
        df_with_noise[col] = df_with_noise[col].fillna('')

# Final dataset statistics
print("\n" + "="*60)
print("Final Dataset Statistics:")
print("="*60)
print(f"Total samples: {len(df_with_noise)}")
print(f"Normal samples: {len(df_with_noise[df_with_noise['attack_type_label'] == 'normal'])}")
print(f"Attack samples: {len(df_with_noise[df_with_noise['attack_type_label'] != 'normal'])}")
print(f"Attack types distribution:\n{df_with_noise['attack_type_label'].value_counts()}")
print(f"Label noise introduced: {noise_ratio*100:.1f}%")
print(f"Missing values present: Yes ({missing_rate*100:.1f}% missing in selected columns)")

# Save to CSV
df_with_noise.to_csv('realtraffic.csv', index=False)
print(f"\nDataset saved to 'realtraffic.csv'")

# Display sample of the transformed dataset
print("\nSample of transformed dataset:")
print(df_with_noise.head(10)[['timestamp', 'source_ip', 'url', 'attack_type_label', 'User-Agent', 'is_attack']])