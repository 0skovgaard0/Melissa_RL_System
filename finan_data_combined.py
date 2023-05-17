


import pandas as pd

# Load and merge the data
reits_data = pd.read_csv("normalized_merged_reits_data_2_0.csv")
sp500_data = pd.read_csv("normalized_merged_SP500_data_2_0.csv")
singlecom_data = pd.read_csv("normalized_merged_singlecom_2_0.csv")
intin_data = pd.read_csv("normalized_merged_intin_data_2_0.csv")
eftcom_data = pd.read_csv("normalized_eftcom_2_0.csv")
bond_data = pd.read_csv("normalized_merged_bond_data_2_0.csv")
crypto_data = pd.read_csv("normalized_crypto_2_0.csv")

merged_data = pd.concat([reits_data, sp500_data, singlecom_data, intin_data, eftcom_data, bond_data, crypto_data], axis=0)

# Save the merged data to a new CSV file
merged_data.to_csv("finan_data_combined.csv", index=False)

print("Merged data:")
print(merged_data.head())