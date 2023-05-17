# # Collecting, Cleaning, and Preprocessing International indicies with Yahoo_fin API and auto update it:

# import pandas as pd
# import datetime
# from yahoo_fin.stock_info import get_data
# from exchange import load_yen_exchange_rates, convert_yen_to_usd, load_euro_exchange_rates, convert_euro_to_usd
# from sklearn.preprocessing import MinMaxScaler

# class IntInData:
#     def __init__(self):
#         pass

#     def fetch_data(self, ticker, start_date, end_date):
#         return get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=True, interval="1d")

#     def clean_data(self, data, ticker):
#         data = data[['open', 'adjclose', 'volume']]
#         data = data.dropna()
#         data.index = pd.to_datetime(data.index)

#         if ticker in ['^N300', '^N225']:
#             exchange_rates = load_yen_exchange_rates(start_date='1971-01-04', end_date='2023-10-04', csv_file='ML_Project_1/yen_usd_FRED.csv')
#             data[['open', 'adjclose', 'volume']] = convert_yen_to_usd(data[['open', 'adjclose', 'volume']], exchange_rates)
#         elif ticker == '^STOXX':
#             exchange_rates = load_euro_exchange_rates(start_date='1999-01-04', end_date='2023-10-04', csv_file='ML_Project_1/ECB_euro_to_usd.csv')
#             data_usd = convert_euro_to_usd(data[['open', 'adjclose', 'volume']], exchange_rates)
#             data[['open', 'adjclose', 'volume']] = data_usd[['open', 'adjclose', 'volume']]

#         data.columns = [f"{ticker}_{col}" for col in data.columns]

#         return data

#     def normalize_data(self, data):
#         scaler = MinMaxScaler()
#         data_scaled = scaler.fit_transform(data)
#         data_normalized = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
#         return data_normalized

#     def merge_data(self, data_list):
#         merged_data = pd.concat(data_list, axis=1)

#         return merged_data

# if __name__ == "__main__":
#     intin_data = IntInData()

#     end_date = "10/04/2023"
#     tickers_intin = ["MSCI", "EEM", "VEU", "IOO", "DGT", "^N300", "^N225", "^STOXX"]

#     cleaned_data_list = []

#     for ticker in tickers_intin:
#         if ticker == '^N225':
#             start_date = "01/05/1971"
#         else:
#             start_date = "01/01/1970"

#         data = intin_data.fetch_data(ticker, start_date, end_date)
#         cleaned_data = intin_data.clean_data(data, ticker)
#         cleaned_data_list.append(cleaned_data)

#     merged_data = intin_data.merge_data(cleaned_data_list)

#     # Normalize the merged data
#     normalized_data = intin_data.normalize_data(merged_data)

#     # Save the normalized merged data to a CSV file
#     normalized_data.to_csv("normalized_merged_intin_data.csv", index_label='date')

#     print("\nNormalized merged data:")
#     print(normalized_data.head())




import pandas as pd
import datetime
from yahoo_fin.stock_info import get_data
from exchange import load_yen_exchange_rates, convert_yen_to_usd, load_euro_exchange_rates, convert_euro_to_usd
from sklearn.preprocessing import MinMaxScaler

class IntInData:
    def __init__(self):
        pass

    def fetch_data(self, ticker, start_date, end_date):
        return get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=True, interval="1d")

    def clean_data(self, data, ticker):
        data = data[['open', 'adjclose', 'volume']]
        data = data.dropna()
        data.index = pd.to_datetime(data.index)

        if ticker in ['^N300', '^N225']:
            exchange_rates = load_yen_exchange_rates(start_date='1971-01-04', end_date='2023-10-04', csv_file='ML_Project_1/yen_usd_FRED.csv')
            data[['open', 'adjclose', 'volume']] = convert_yen_to_usd(data[['open', 'adjclose', 'volume']], exchange_rates)
        elif ticker == '^STOXX':
            exchange_rates = load_euro_exchange_rates(start_date='1999-01-04', end_date='2023-10-04', csv_file='ML_Project_1/ECB_euro_to_usd.csv')
            data_usd = convert_euro_to_usd(data[['open', 'adjclose', 'volume']], exchange_rates)
            data[['open', 'adjclose', 'volume']] = data_usd[['open', 'adjclose', 'volume']]

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
        long_format_data.to_csv("normalized_merged_intin_data_2_0.csv", index=False)
     
        return long_format_data
    
    def extract_stoxx_data(self, start_date, end_date, source_csv='normalized_merged_intin_data_2_0.csv', dest_csv='stoxx_recent_data.csv'):
        # Load data from the CSV file
        df = pd.read_csv(source_csv)

        # Convert the 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Set 'Date' as the index
        df.set_index('Date', inplace=True)

        # Filter rows by date range and asset name
        stoxx_data = df[(df.index >= start_date) & (df.index <= end_date) & (df['Asset name'] == '^STOXX')]

        # Save the filtered data to a new CSV file
        stoxx_data.to_csv(dest_csv)

        return stoxx_data

if __name__ == "__main__":
    intin_data = IntInData()

    end_date = "10/05/2023"
    tickers_intin = ["MSCI", "EEM", "VEU", "IOO", "DGT", "^N300", "^N225", "^STOXX"]

    cleaned_data_list = []

    for ticker in tickers_intin:
        if ticker == '^N225':
            start_date = "01/05/1971"
        else:
            start_date = "01/01/1970"

        data = intin_data.fetch_data(ticker, start_date, end_date)
        cleaned_data = intin_data.clean_data(data, ticker)
        cleaned_data_list.append(cleaned_data)

    # Merge and fill data
    merged_data = intin_data.merge_and_fill_data(cleaned_data_list)

    # Print the merged data
    print("\nNormalized merged data:")
    print(merged_data.head())
