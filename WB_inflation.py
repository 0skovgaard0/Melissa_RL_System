# # We will import the WB_inflation.CSV file and modify the date to be useful for our Melissa's RL Machine project:
# # Let's use wbgapi to collect Inflation, consumer prices (annual %) (FP.CPI.TOTL.ZG) data from The World Bank:


# import wbgapi as wb
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import SelectKBest, f_regression


# import wbgapi as wb
# import pandas as pd

# class WB_Inflation_Data:
#     def __init__(self):
#         pass

#     def fetch_wb_data(self, series, countries, years):
#         """
#         Fetches inflation data from World Bank API and returns a DataFrame
        
#         Args:
#         - series (list of str): List of series codes to fetch from the API
#         - countries (list of str): List of country codes to fetch data for
#         - years (range): Range of years to fetch data for
        
#         Returns:
#         - inflation_raw_data (DataFrame): DataFrame containing raw inflation data for specified countries and years
#         """
#         # Fetch data from API
#         data = wb.data.DataFrame(series, countries, time=years)
        
#         # Clean data by flattening the multi-level columns and resetting the index
#         data.columns = [col[0] for col in data.columns]
#         data = data.reset_index()
        
#         # Insert the 'Series' column with the series code and rename columns
#         inflation_raw_data = pd.DataFrame(data)
#         inflation_raw_data.insert(1, 'Series', series[0])
#         inflation_raw_data.columns = ['Country code', 'Series'] + [str(year) for year in inflation_raw_data.columns[2:]]

#         return inflation_raw_data


#     def clean_and_transform_data(self, data):
#         """
#         Cleans and transforms raw inflation data
        
#         Args:
#         - data (DataFrame): DataFrame containing raw inflation data
        
#         Returns:
#         - cleaned_data (DataFrame): DataFrame containing cleaned and transformed inflation data
#         """
#         # Drop rows with NaN values in the 'Series' column and drop the 'Series' column
#         data = data.dropna(subset=['Series'])
#         data = data.drop(columns=['Series'])

#         # Melt the DataFrame to long format and extract year from the date string
#         data = data.melt(id_vars=['Country code'], var_name='date', value_name='value')
#         data['date'] = data['date'].str.extract('(\d+)')
        
#         # Drop rows with NaN values in the 'date' column and convert the 'date' column to integer
#         data = data.dropna(subset=['date'])
#         data['date'] = data['date'].astype(int)
        
#         # Pivot the DataFrame to wide format and save to a new CSV file
#         cleaned_data = data.pivot_table(values='value', index=['Country code'], columns='date').reset_index()
#         cleaned_data.to_csv('inflation_updated_data.csv', index=False)

#         return cleaned_data
    

# if __name__ == "__main__":
#     # Initialize class
#     wb_inflation_data = WB_Inflation_Data()

#     # Set parameters for fetching inflation data from World Bank API
#     series = ['FP.CPI.TOTL.ZG']
#     countries = ['BRA', 'CAN', 'CHN', 'FRA', 'DEU', 'IND', 'ITA', 'JPN', 'GBR', 'USA']
#     years = range(1970, 2023)

#     # Fetch raw inflation data and save to CSV file
#     inflation_raw_data = wb_inflation_data.fetch_wb_data(series, countries, years)
#     inflation_raw_data.to_csv("inflation_raw_data.csv", index=False)

#     # Read CSV file, rename columns, and save to new CSV file
#     inflation_df = pd.read_csv('inflation_raw_data.csv', parse_dates=True)
#     inflation_df.columns = ['Country code', 'Series'] + [str(year) for year in range(1970, 2022)]
#     inflation_df.to_csv('WB_inflation_preprocessed.csv', index=False)

#     inflation_clean_data = wb_inflation_data.clean_and_transform_data(inflation_raw_data)


# import wbgapi as wb
# import pandas as pd

# class WB_Inflation_Data:
#     def __init__(self):
#         pass

#     def fetch_wb_data(self, series, countries, years):
#         """
#         Fetches inflation data from World Bank API and returns a DataFrame
        
#         Args:
#         - series (list of str): List of series codes to fetch from the API
#         - countries (list of str): List of country codes to fetch data for
#         - years (range): Range of years to fetch data for
        
#         Returns:
#         - inflation_raw_data (DataFrame): DataFrame containing raw inflation data for specified countries and years
#         """
#         # Fetch data from API
#         data = wb.data.DataFrame(series, countries, time=years)
        
#         # Clean data by flattening the multi-level columns and resetting the index
#         data.columns = [col[0] for col in data.columns]
#         data = data.reset_index()
        
#         # Insert the 'Series' column with the series code and rename columns
#         inflation_raw_data = pd.DataFrame(data)
#         inflation_raw_data.insert(1, 'Series', series[0])
#         inflation_raw_data.columns = ['Country code', 'Series'] + [str(year) for year in inflation_raw_data.columns[2:]]

#         return inflation_raw_data


#     def clean_and_transform_data(self, data):
#         """
#         Cleans and transforms raw inflation data
        
#         Args:
#         - data (DataFrame): DataFrame containing raw inflation data
        
#         Returns:
#         - cleaned_data (DataFrame): DataFrame containing cleaned and transformed inflation data
#         """
#         # Drop rows with NaN values in the 'Series' column and drop the 'Series' column
#         data = data.dropna(subset=['Series'])
#         data = data.drop(columns=['Series'])

#         # Melt the DataFrame to long format and extract year from the date string
#         data = data.melt(id_vars=['Country code'], var_name='date', value_name='value')
#         data['date'] = data['date'].str.extract('(\d+)')
        
#         # Drop rows with NaN values in the 'date' column and convert the 'date' column to integer
#         data = data.dropna(subset=['date'])
#         data['date'] = data['date'].astype(int)
        
#         # Pivot the DataFrame to wide format and save to a new CSV file
#         cleaned_data = data.pivot_table(values='value', index=['Country code'], columns='date').reset_index()
#         cleaned_data.to_csv('inflation_updated_data.csv', index=False)

#         return cleaned_data
    

# if __name__ == "__main__":
#     # Initialize class
#     wb_inflation_data = WB_Inflation_Data()

#     # Set parameters for fetching inflation data from World Bank API
#     series = ['FP.CPI.TOTL.ZG']
#     countries = ['BRA', 'CAN', 'CHN', 'FRA', 'DEU', 'IND', 'ITA', 'JPN', 'GBR', 'USA']
#     years = range(1970, 2023)

#     # Fetch raw inflation data and save to CSV file
#     inflation_raw_data = wb_inflation_data.fetch_wb_data(series, countries, years)
#     inflation_raw_data.to_csv("inflation_raw_data.csv", index=False)

#     # Read CSV file, rename columns, and save to new CSV file
#     inflation_df = pd.read_csv('inflation_raw_data.csv', parse_dates=True)
#     inflation_df.columns = ['Country code', 'Series'] + [str(year) for year in range(1970, 2022)]
#     inflation_df.to_csv('WB_inflation_preprocessed.csv', index=False)

#     # Clean and transform the data
#     inflation_clean_data = wb_inflation_data.clean_and_transform_data(inflation_raw_data)


import wbgapi as wb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class WB_Inflation_Data:
    def __init__(self):
        pass

    def fetch_wb_data(self, series, countries, years):
        # Fetch data from API
        data = wb.data.DataFrame(series, countries, time=years)
        
        # Clean data by flattening the multi-level columns and resetting the index
        data.columns = [col[0] for col in data.columns]
        data = data.reset_index()
        
        # Insert the 'Series' column with the series code and rename columns
        inflation_raw_data = pd.DataFrame(data)
        inflation_raw_data.insert(1, 'Series', series[0])
        inflation_raw_data.columns = ['Country code', 'Series'] + [str(year) for year in inflation_raw_data.columns[2:]]

        return inflation_raw_data


    def clean_and_transform_data(self, data):
        # Drop rows with NaN values in the 'Series' column and drop the 'Series' column
        data = data.dropna(subset=['Series'])
        data = data.drop(columns=['Series'])

        # Melt the DataFrame to long format and extract year from the date string
        data = data.melt(id_vars=['Country code'], var_name='date', value_name='value')
        data['date'] = data['date'].str.extract('(\d+)')
        
        # Drop rows with NaN values in the 'date' column and convert the 'date' column to integer
        data = data.dropna(subset=['date'])
        data['date'] = data['date'].astype(int)
        
        # Pivot the DataFrame to wide format and save to a new CSV file
        cleaned_data = data.pivot_table(values='value', index=['Country code'], columns='date').reset_index()
        cleaned_data.to_csv('inflation_updated_data.csv', index=False)

        return cleaned_data
    

if __name__ == "__main__":
    # Initialize class
    wb_inflation_data = WB_Inflation_Data()

    # Set parameters for fetching inflation data from World Bank API
    series = ['FP.CPI.TOTL.ZG']
    countries = ['BRA', 'CAN', 'CHN', 'FRA', 'DEU', 'IND', 'ITA', 'JPN', 'GBR', 'USA']
    years = range(1970, 2023)

    # Fetch raw inflation data and save to CSV file
    inflation_raw_data = wb_inflation_data.fetch_wb_data(series, countries, years)
    inflation_raw_data.to_csv("inflation_raw_data.csv", index=False)

    # Read CSV file, rename columns, and save to new CSV file
    inflation_df = pd.read_csv('inflation_raw_data.csv', parse_dates=True)
    inflation_df.columns = ['Country code', 'Series'] + [str(year) for year in range(1970, 2022)]
    inflation_df.to_csv('WB_inflation_preprocessed.csv', index=False)

    # Clean and transform the data
    inflation_clean_data = wb_inflation_data.clean_and_transform_data(inflation_raw_data)

    # Make a 2.0 version, long format to merge with the other Macroeconomic data files
    df = pd.read_csv('WB_inflation_preprocessed.csv')

    # rename the 'Country code' column to 'Asset name' to match the other files for merging later
    df.rename(columns={'Country code': 'Asset name'}, inplace=True)

    # specify the columns that should remain as identifiers
    id_vars = ['Asset name', 'Series']

    # melt the DataFrame to long format, using 'Year' columns as variable names
    df_long = pd.melt(df, id_vars=id_vars, var_name='Year', value_name='Value')

    # convert 'Year' column to datetime format
    df_long['Date'] = pd.to_datetime(df_long['Year'], format='%Y') + pd.offsets.YearEnd(0)

    # drop the original 'Year' column and any rows with missing values in 'Value' column
    df_long = df_long.drop(['Year'], axis=1).dropna(subset=['Value'])

    # Normalize the 'Value' column using MinMax
    scaler = MinMaxScaler()
    df_long['Value'] = scaler.fit_transform(df_long[['Value']])

    # reorder the columns and save to a new CSV 2.0 file
    df_long = df_long[['Date', 'Asset name', 'Series', 'Value']]
    df_long.to_csv('WB_inflation_preprocessed_2_0.csv', index=False)












  