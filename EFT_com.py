# Collecting, Cleaning, and Preprocessing EFT Commodities with Yahoo_fin API and auto update it:

# import pandas as pd
# import datetime
# from yahoo_fin.stock_info import get_data
# from sklearn.preprocessing import MinMaxScaler

# class EFTComData:
#     def __init__(self):
#         pass

#     def fetch_eftcom_data(self, ticker, start_date, end_date):
#         return get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=True, interval="1d")

#     def clean_data(self, data, ticker):
#         # Select the required columns
#         data = data[['open', 'adjclose', 'volume']]

#         # Drop rows with missing values
#         data = data.dropna()

#         # Convert the date column to a datetime object
#         data.index = pd.to_datetime(data.index)

#         # Rename columns with the ticker name
#         data.columns = [f"{ticker}_{col}" for col in data.columns]

#         return data

#     def normalize_data(self, data):
#         scaler = MinMaxScaler()
#         data_scaled = scaler.fit_transform(data)
#         data_normalized = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
#         return data_normalized

#     def merge_and_fill_data(self, data_list):
#         # Merge data
#         merged_data = pd.concat(data_list, axis=1)

#         # Normalize data
#         normalized_data = self.normalize_data(merged_data)

#         # Save merged normalized data to output file
#         normalized_data.to_csv("normalized_merged_eftcom_data.csv", index_label='date')

#         return normalized_data

# if __name__ == "__main__":
#     eftcom_data = EFTComData()

#     start_date = "01/01/1970"
#     end_date = "10/04/2023"

#     tickers_eftcom = ["USO", "GLD", "SLV", "DBB", "DBA", "GSG"]

#     cleaned_data_list = []

#     # Clean data for all tickers
#     for ticker in tickers_eftcom:
#         data = eftcom_data.fetch_eftcom_data(ticker, start_date, end_date)
#         cleaned_data = eftcom_data.clean_data(data, ticker)
#         cleaned_data_list.append(cleaned_data)

#     # Merge and fill data
#     merged_data = eftcom_data.merge_and_fill_data(cleaned_data_list)

#     # Print the merged data
#     print("\nNormalized merged data:")
#     print(merged_data.head())


import pandas as pd
import datetime
from yahoo_fin.stock_info import get_data
from sklearn.preprocessing import MinMaxScaler

class EFTComData:
    def __init__(self):
        pass

    def fetch_eftcom_data(self, ticker, start_date, end_date):
        return get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=True, interval="1d")

    def clean_data(self, data, ticker):
        # Select the required columns
        data = data[['open', 'adjclose', 'volume']]

        # Drop rows with missing values
        data = data.dropna()

        # Convert the date column to a datetime object
        data.index = pd.to_datetime(data.index)

        # Rename columns with the ticker name
        data.columns = [f"{ticker}_{col}" for col in data.columns]

        return data

    def normalize_data(self, data):
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        data_normalized = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
        data_normalized.index.name = 'Date'
        return data_normalized

    def merge_and_fill_data(self, data_list):
        # Merge data
        merged_data = pd.concat(data_list, axis=1)

        # Normalize data
        normalized_data = self.normalize_data(merged_data)

        # Melt the normalized data to long format
        id_vars = ['Date']
        long_format_data = pd.melt(normalized_data.reset_index(), id_vars=id_vars, var_name='Series', value_name='Value')

        # Drop rows with missing values
        long_format_data = long_format_data.dropna(subset=['Value'])

        # Split the 'Series' column into 'Asset name' and 'Series'
        long_format_data[['Asset name', 'Series']] = long_format_data['Series'].str.split('_', expand=True)

        # Rearrange columns
        long_format_data = long_format_data[['Date', 'Asset name', 'Series', 'Value']]

        # Sort data by Date
        long_format_data = long_format_data.sort_values('Date')

        # Save the long format data to a new CSV file
        long_format_data.to_csv("normalized_eftcom_2_0.csv", index=False)

        return long_format_data

if __name__ == "__main__":
    eftcom_data = EFTComData()

    start_date = "01/01/1970"
    end_date = "10/04/2023"

    tickers_eftcom = ["USO", "GLD", "SLV", "DBB", "DBA", "GSG"]

    cleaned_data_list = []

    # Clean data for all tickers
    for ticker in tickers_eftcom:
        data = eftcom_data.fetch_eftcom_data(ticker, start_date, end_date)
        cleaned_data = eftcom_data.clean_data(data, ticker)
        cleaned_data_list.append(cleaned_data)

    # Merge and fill data
    merged_data = eftcom_data.merge_and_fill_data(cleaned_data_list)

    # Print the merged data
    print("\nnormalized EFT Commodities 2.0 Data:")
    print(merged_data.head())

# USO (crude oil futures) data from Yahoo Finance:
# GLD (Gold)
# SLV (Silver)
# DBB (industrial metals, such as aluminum, copper, and zinc)
# DBA (agricultural commodities, such as corn, soybeans, wheat, and sugar)
# GSG (commodities across energy, metals, and agriculture sectors)



