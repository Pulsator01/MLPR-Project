import pandas as pd

# Read the processed data
df = pd.read_csv('processed_data.csv')

# Split based on Invoice Quantity
high_quantity = df[df['Invoice Quantity'] > 30]
low_quantity = df[df['Invoice Quantity'] <= 30]

# Save to new CSV files
high_quantity.to_csv('invoice_quantity_over_30.csv', index=False)
low_quantity.to_csv('invoice_quantity_below_30.csv', index=False)

# Print statistics
print(f'Original data: {len(df)} rows')
print(f'Invoice Quantity > 30: {len(high_quantity)} rows')
print(f'Invoice Quantity <= 30: {len(low_quantity)} rows')