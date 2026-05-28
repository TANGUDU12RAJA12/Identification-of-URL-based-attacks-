import pandas as pd
df = pd.read_csv('cyber_attack_dataset_final.csv')

# Check data types
#print(df.dtypes)

# Save to CSV with minimal quoting (only when necessary)
#df.to_csv('url_attack_dataset.csv', index=False, quoting=0)  # QUOTE_MINIMAL

# Or simply use default (same effect)
# df.to_csv('url_attack_dataset.csv', index=False)

print(f"Shape: {df.shape}")
print(f"\nRows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"\nColumn Names:\n{list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")