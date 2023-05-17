# # Collecting, Cleaning, and Preprocessing Bonds with Yahoo_fin API and auto update it:

from fredapi import Fred
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler

class EmploymentData:
    def __init__(self, api_key):
        self.api_key = api_key
        self.fred = Fred(api_key=self.api_key)
        self.series_ids = ["PAYEMS", "CES9091000001", "USTRADE", "UMCSENT", "UNRATE", "ICSA", "M1SL", "SP500", "VIXCLS"]

    def fetch_data(self):
        start_date = datetime.date(1970, 1, 1)
        end_date = datetime.date.today()
        data = {}
        for series_id in self.series_ids:
            series_data = pd.DataFrame(self.fred.get_series(series_id, start_date, end_date))
            series_data.columns = [series_id]
            data[series_id] = series_data
        return pd.concat(data.values(), axis=1, keys=data.keys()).droplevel(0, axis=1)

    def clean_and_transform_data(self, data):
        # Melt the DataFrame to long format and extract year from the date string
        data = data.reset_index()
        data = data.melt(id_vars=['index'], value_name='value')
        data['date'] = pd.to_datetime(data['index'])
        data['year'] = data['date'].dt.year
        
        # Rename columns
        data = data.rename(columns={'variable': 'Asset name', 'index': 'Date'})
        data = data.drop(columns=['date'])
        
        # Pivot the DataFrame to wide format and save to a new CSV file
        cleaned_data = data.pivot_table(values='value', index=['Date', 'Asset name'], columns=None).reset_index()
        # cleaned_data.columns = cleaned_data.columns.str.replace(' ', '_')
        cleaned_data.to_csv('employ_rate_updated_data.csv', index=False)

        return cleaned_data

if __name__ == "__main__":
    # Set API key and create a EmploymentData instance
    api_key = "af08e5017b1a9d6a033648512f73c60c"
    employ_data = EmploymentData(api_key)

    # Fetch raw employ rate data and save to CSV file
    employ_raw_data = employ_data.fetch_data()
    employ_raw_data.to_csv("employ_raw_data.csv", index_label='date')

    # Clean and transform the data
    employ_clean_data = employ_data.clean_and_transform_data(employ_raw_data)

    # Save the preprocessed data to a new CSV file
    employ_clean_data.to_csv('employ_rate_preprocessed.csv', index=False)
    
    # Make a 2.0 version, long format to merge with the other Macroeconomic data files
    df = pd.read_csv('employ_rate_preprocessed.csv')

    # specify the columns that should remain as identifiers
    id_vars = ['Date', 'Asset name']

    # melt the DataFrame to long format, using 'Year' columns as variable names
    df_long = pd.melt(df, id_vars=id_vars, var_name='Year', value_name='Value')

    # convert 'Year' column to datetime format
    df_long['Date'] = pd.to_datetime(df_long['Date'], format='%Y-%m-%d') + pd.offsets.YearEnd(0)

    # drop the original 'Year' column and any rows with missing values in 'Value' column
    df_long = df_long.drop(['Year'], axis=1).dropna(subset=['Value'])

    # Normalize the 'Value' column using MinMax
    scaler = MinMaxScaler()
    df_long['Value'] = scaler.fit_transform(df_long[['Value']])

    # reorder the columns and save to a new CSV file
    df_long['Series'] = df_long['Asset name']
    df_long = df_long[['Date', 'Asset name', 'Series', 'Value']]

    df_long.to_csv('employ_rate_preprocessed_2_0.csv', index=False)


    
