# Implementation of the exchange rate functions for Yen and Euro to USD:
# Euro to USD data comes from the European Central Bank (ECB) and Yahoo Finance
# Yen to USD data comes from the Federal Reserve Bank of St. Louis (FRED) and Yahoo Finance

import pandas as pd
from yahoo_fin.stock_info import get_data


# Yen to USD exchange rate data:
def load_yen_exchange_rates(start_date, end_date, csv_file='ML_Project_1/yen_usd_FRED.csv'):
    exchange_rates = pd.read_csv(csv_file, index_col='Date', parse_dates=True)

    start_date = max(exchange_rates.index[-1] + pd.Timedelta(days=1), pd.Timestamp(start_date))
    end_date = pd.Timestamp(end_date)

    if start_date <= end_date:
        yen_usd = get_data('JPY=X', start_date=start_date, end_date=end_date, index_as_date=True, interval="1d")
        exchange_rates = pd.concat([exchange_rates, yen_usd])

    return exchange_rates

def convert_yen_to_usd(yen_data, exchange_rates):
    yen_usd = yen_data.copy()
    
    # Rename the columns in the exchange_rates DataFrame to avoid conflicts during the join operation
    exchange_rates = exchange_rates.rename(columns={'open': 'usd_open', 'adjclose': 'usd_adjclose', 'volume': 'usd_volume'})
    
    yen_usd = yen_usd.join(exchange_rates['usd_adjclose'], how='left')
    yen_usd['usd_adjclose'].fillna(method='ffill', inplace=True)
    yen_usd['usd_adjclose'].fillna(method='bfill', inplace=True)

    for col in ['open', 'adjclose']:
        yen_usd[col] = yen_usd[col] / yen_usd['usd_adjclose']

    yen_usd.drop(columns=['usd_adjclose'], inplace=True)

    return yen_usd



# Euro to USD exchange rate data:
def load_euro_exchange_rates(start_date, end_date, csv_file='ML_Project_1/ECB_euro_to_usd.csv'):
    exchange_rates = pd.read_csv(csv_file, index_col='DATE', parse_dates=True)

    start_date = max(exchange_rates.index[-1] + pd.Timedelta(days=1), pd.Timestamp(start_date))

    if pd.Timestamp(start_date) <= pd.Timestamp(end_date):
        eur_usd = get_data('EURUSD=X', start_date=start_date, end_date=end_date)
        exchange_rates = pd.concat([exchange_rates, eur_usd])

    return exchange_rates

def convert_euro_to_usd(euro_data, exchange_rates):
    data_usd = euro_data.copy()
    data_usd = data_usd.merge(exchange_rates.add_suffix('_exchange'), left_index=True, right_index=True, how='left')
    data_usd['adjclose_exchange'] = data_usd['adjclose_exchange'].interpolate(method='time')

    for col in ['open', 'adjclose']:
        data_usd[col] = data_usd[col] * data_usd['adjclose_exchange']

    # Drop unnecessary columns
    data_usd.drop(columns=[col for col in data_usd.columns if col.endswith('_exchange')], inplace=True)

    return data_usd