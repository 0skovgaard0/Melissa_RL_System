import pandas as pd
from yahoo_fin.stock_info import get_data
from sklearn.preprocessing import MinMaxScaler

class SP500Data:
    def __init__(self):
        pass

    def fetch_data(self, ticker, start_date, end_date):
        return get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=True, interval="1d")

    def clean_data(self, data, ticker):
        data = data[['open', 'adjclose', 'volume']]
        data = data.dropna()
        data.index = pd.to_datetime(data.index)

        # Update 'open' value if it's 0 and previous 'adjclose' value > 0
        data.loc[(data['open'] == 0) & (data['adjclose'].shift(1) > 0), 'open'] = data['adjclose'].shift(1)

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
        long_format_data.to_csv("normalized_merged_SP500_data_2_0.csv", index=False)

        return long_format_data

if __name__ == "__main__":
    sp500_data = SP500Data()

    start_date = "01/01/1970"
    end_date = "10/04/2023"

    tickers = ["^GSPC", "^RUT", "^IXIC"]

    cleaned_data_list = []

    for ticker in tickers:
        data = sp500_data.fetch_data(ticker, start_date, end_date)
        cleaned_data = sp500_data.clean_data(data, ticker)
        cleaned_data_list.append(cleaned_data)

    merged_data = sp500_data.merge_and_fill_data(cleaned_data_list)

    print("\nNormalized merged data:")
    print(merged_data.head())

