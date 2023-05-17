# Purpose of Merging Data Sets
# Here we merge multiple preprocessed financial and macroeconomic data 
# sets into a single dataset that will be used for training and evaluation of 
# the Melissa Project's Hybrid Neural Network. 
# The merged dataset will be used to identify asset classes and investment options, 
# analyze historical data and economic factors, incorporate macroeconomic factors, 
# and optimize the portfolio using deep reinforcement learning 
# while adhering to a $1,000,000 budget constraint.


# import pandas as pd
import pandas as pd

# Read in the preprocessed CSV files for each dataset
df_list_financial = []
df_list_macro = []

# Financial Data
df_reits = pd.read_csv('normalized_merged_reits_data_2_0.csv')
df_sp500 = pd.read_csv('normalized_merged_SP500_data_2_0.csv')
df_singlecom = pd.read_csv('normalized_merged_singlecom_2_0.csv')
df_intin = pd.read_csv('normalized_merged_intin_data_2_0.csv')
df_eftcom = pd.read_csv('normalized_eftcom_2_0.csv')
df_bond = pd.read_csv('normalized_merged_bond_data_2_0.csv')
df_crypto = pd.read_csv('normalized_crypto_2_0.csv')

# Append each financial dataset to the list
df_list_financial.extend([df_reits, df_sp500, df_singlecom, df_intin, df_eftcom, df_bond, df_crypto])

# Macroeconomic Data
df_bnk = pd.read_csv('bnk_int_rate_preprocessed_2_0.csv')
df_emp = pd.read_csv('employ_rate_preprocessed_2_0.csv')
df_gdp = pd.read_csv('WB_GDP_preprocessed_2_0.csv')
df_inf = pd.read_csv('WB_inflation_preprocessed_2_0.csv')

# Append each macroeconomic dataset to the list
df_list_macro.extend([df_bnk, df_emp, df_gdp, df_inf])

# Merge all the datasets in the financial list into a single dataset
df_merged_financial = pd.concat(df_list_financial, axis=0)

# Merge all the datasets in the macroeconomic list into a single dataset
df_merged_macro = pd.concat(df_list_macro, axis=0)

print("Financial Data Shape:", df_merged_financial.shape)
print("Macroeconomic Data Shape:", df_merged_macro.shape)


# print amount of missing data by column and NaN values in each column
print("Financial Data Missing Values:\n", df_merged_financial.isnull().sum())
print("Macroeconomic Data Missing Values:\n", df_merged_macro.isnull().sum())

# Convert the 'Date' column in df_merged_financial to datetime format
df_merged_financial['Date'] = pd.to_datetime(df_merged_financial['Date'])

# Define the cutoff date
cutoff_date = pd.to_datetime("2023-04-10")

# Filter the data
filtered_data = df_merged_financial[df_merged_financial['Date'] <= cutoff_date]

# Save the filtered data to the CSV file
filtered_data.to_csv('merged_financial_data.csv', index=False)

# Save the merged macroeconomic dataset to a CSV file
df_merged_macro.to_csv('merged_macroeconomic_data.csv', index=False)



# Print a list of Unique 'Asset name' values from the merged financial dataset
print("Unique 'Asset name' values from the merged financial dataset:\n", df_merged_financial['Asset name'].unique())