import pandas as pd
from INT_in import IntInData, convert_yen_to_usd
from exchange import load_yen_exchange_rates
from yahoo_fin.stock_info import get_data
from sklearn.preprocessing import MinMaxScaler
from feature_extraction import DataPreprocessor

class DataAggregator:
    def __init__(self, window_size):
        self.window_size = window_size
        self.intin = IntInData()
        self.data_preprocessor = DataPreprocessor(window_size)

    def clean_data(self, data, ticker):
        data = data[['open', 'adjclose', 'volume']]
        data = data.dropna()
        data.index = pd.to_datetime(data.index)
        data.columns = [f"{ticker}_{col}" for col in data.columns]
        return data

    def normalize_data(self, data):
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        data_normalized = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)
        data_normalized.index.name = 'Date'
        return data_normalized

    def long_format(self, data, asset_name):
        data = data.stack().reset_index()
        data.columns = ['Date', 'Series', 'Value']
        data['Asset name'] = asset_name
        return data

    def collect_data(self, ticker, start_date, end_date):
        asset_name = ticker
        if ticker in ['^N300', '^N225']:
            data = self.intin.fetch_data(ticker, start_date, end_date)
            data = self.intin.clean_data(data, ticker)
            exchange_rates = load_yen_exchange_rates(start_date='1971-01-04', end_date='2023-05-10', csv_file='ML_Project_1/yen_usd_FRED.csv')
            required_columns = set(['open', 'adjclose', 'volume'])
            if required_columns.issubset(data.columns):
                data[['open', 'adjclose', 'volume']] = convert_yen_to_usd(data[['open', 'adjclose', 'volume']], exchange_rates)
                data = self.normalize_data(data)  # normalize the data for '^N300', '^N225'
            else:
                missing_columns = required_columns - set(data.columns)
                print(f"Some of the required columns are missing in the DataFrame. Missing columns: {missing_columns}")

        elif ticker == '^STOXX':
            data = self.intin.extract_stoxx_data(start_date, end_date)
            data = data.pivot_table(index='Date', columns='Series', values='Value')
            data.columns = [f"{ticker}_{col}" for col in data.columns]
        else:
            data = get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=True, interval="1d")
            data = self.clean_data(data, ticker)
            data = self.normalize_data(data)
        data = self.long_format(data, asset_name)  # convert to long format
        return data

    def aggregate_data(self, tickers, start_date, end_date):
        data_list = []
        for ticker in tickers:
            data = self.collect_data(ticker, start_date, end_date)
            data_list.append(data)
        aggregated_data = pd.concat(data_list, axis=0)  # aggregate long format data
        aggregated_data = aggregated_data.dropna(subset=['Value'])
        aggregated_data[['Asset name', 'Series']] = aggregated_data['Series'].str.split('_', expand=True)
        aggregated_data = aggregated_data[['Date', 'Asset name', 'Series', 'Value']]
        aggregated_data = aggregated_data.sort_values('Date')
        return aggregated_data

    def save_to_csv(self, df, filename):
        df.to_csv(filename, index=False)

    def preprocess_and_save_features(self, filename):
        # Load data
        new_data = pd.read_csv(filename)

        # Filter to keep only the 'adjclose' series
        recent_data = new_data[new_data['Series'] == 'adjclose']

        # Preprocess data
        recent_data = self.data_preprocessor.preprocess_data(recent_data.copy())

        # Save the processed features to a new CSV file
        recent_data.to_csv('recent_features_financial.csv', index=False)

        return recent_data

    def prepare_data_for_rl(self, recent_data, historical_data):
        # Filter recent data to include only 'adjclose' series
        recent_data = recent_data[recent_data['Series'] == 'adjclose']

        # Print the column names in recent_data
        print("Column names in recent_data:")
        print(recent_data.columns)

        # Forward-fill missing values in price_updates
        price_updates = recent_data.pivot(index='Date', columns='Asset name', values='Value').diff().iloc[1:]
        price_updates = price_updates.ffill().bfill()

        # Combine recent and historical data
        combined_data = pd.concat([historical_data, recent_data])

        # Drop duplicates
        combined_data = combined_data.drop_duplicates(subset=['Date', 'Asset name'])

        # Pivot the data: dates as index, asset names as columns, and 'Value' as values
        combined_data_pivot = combined_data.pivot(index='Date', columns='Asset name', values='Value')

        # The initial prices are the first non-NaN prices for each asset
        initial_prices = combined_data_pivot.apply(lambda series: series.dropna().iloc[0])

        # Keep only the common assets in 'initial_prices' and 'price_updates'
        common_assets = initial_prices.index.intersection(price_updates.columns)
        initial_prices = initial_prices[common_assets]
        price_updates = price_updates[common_assets]

        print("Price Updates DataFrame:")
        print(price_updates.head())
        print("\nInitial prices DataFrame:")
        print(initial_prices.head())
        print("Number of columns in price_updates:", len(price_updates.columns))
        print("Number of columns in initial_prices:", len(initial_prices))

        return initial_prices, price_updates

    


if __name__ == "__main__":
    N = 5  # window_size
    data_aggregator = DataAggregator(N)

    tickers = ['VNO', 'SPG', 'AVB', 'VTR', 'BXP', 'PLD', 'WPC', 'EQIX', 'DLR', '^GSPC', '^IXIC',
               '^RUT', 'CC=F', 'KC=F', 'SB=F', 'ZC=F', 'CL=F', 'GC=F', 'SI=F', 'HG=F', 'ZS=F',
               'ALI=F', '^N225', 'DGT', 'IOO', 'EEM', '^STOXX', '^N300', 'VEU', 'MSCI', 'GLD',
               'USO', 'SLV', 'GSG', 'DBA', 'DBB', 'SHY', 'LQD', 'TLT', 'AGG', 'TIP', 'BNDW',
               'BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD', 'GDLC']

    start_date = "2023-04-11"
    end_date = "2023-05-12"

    new_data = data_aggregator.aggregate_data(tickers, start_date, end_date)
    print("New Data Missing Values:\n", new_data.isnull().sum())
    historical_data = pd.read_csv('combined_data_features.csv')

    data_aggregator.save_to_csv(new_data, 'recent_updated_financial_data.csv')

    # Preprocess the new data and save the features
    recent_data = data_aggregator.preprocess_and_save_features('recent_updated_financial_data.csv')

    initial_prices, price_updates = data_aggregator.prepare_data_for_rl(recent_data, historical_data)
    # print("Historical Data Head:\n", historical_data.head(25))
    # print("\nNew Recent Data Head:\n", new_data.head(25))

    updated_data = pd.concat([historical_data, new_data])
    updated_data.to_csv('updated_combined_data_features.csv', index=False)
    # print("\nUpdated Combined Data Head:\n", updated_data.head(25))



