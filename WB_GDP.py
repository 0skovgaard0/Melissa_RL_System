# import wbgapi as wb
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.feature_selection import SelectKBest, f_regression


import pandas as pd
import wbgapi as wb
from sklearn.preprocessing import MinMaxScaler


class WB_GDP_Data:
    def fetch_wb_gdp_data(self, series, countries, years):
        data = wb.data.DataFrame(series, countries, time=years)
        # Flatten the multi-level columns
        data.columns = [col[0] for col in data.columns]
        
        # Add column names
        data = data.reset_index()
        gdp_raw_data = pd.DataFrame(data)
        gdp_raw_data.columns = ['Country code', 'Series'] + [str(year) for year in gdp_raw_data.columns[2:]]
        return gdp_raw_data

    def clean_and_transform_data(self, data):
        data = data.melt(id_vars=['Country code', 'Series'], var_name='date', value_name='value')
        
        # Extract year from the date string
        data['date'] = data['date'].str.extract('(\d+)')
        
        # Drop rows with NaN values in the 'date' column
        data = data.dropna(subset=['date'])
        
        # Convert the 'date' column to integer
        data['date'] = data['date'].astype(int)
        
        data = data.pivot_table(values='value', index=['Country code', 'Series'], columns='date').reset_index()
        
        # Save the updated dataframe to a new CSV file
        data.to_csv('GDP_updated_data.csv', index=False)
        return data


if __name__ == "__main__":
    wb_gdp_data = WB_GDP_Data()

    series = ['NY.GDP.MKTP.CD', 'NY.GDP.MKTP.KD.ZG']
    countries = ['BRA', 'CAN', 'CHN', 'FRA', 'DEU', 'IND', 'ITA', 'JPN', 'GBR', 'USA']
    years = range(1970, 2023)

    gdp_raw_data = wb_gdp_data.fetch_wb_gdp_data(series, countries, years)
    gdp_raw_data.to_csv("GDP_raw_data.csv", index=False)

    # Read and preprocess the data
    gdp_df = pd.read_csv('GDP_raw_data.csv', parse_dates=True)

    # Modify column names
    gdp_df.columns = ['Country code', 'Series'] + [str(year) for year in range(1970, 2022)]
    
    # Save the updated dataframe to a Preprocessed CSV file
    gdp_df.to_csv('WB_GDP_preprocessed.csv', index=False)

    # Clean and transform the data
    wb_gdp_data.clean_and_transform_data(gdp_raw_data)

    # Make a 2.0 version, long format to merge with the other Macroeconomic data files
    df = pd.read_csv('WB_GDP_preprocessed.csv')
    
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
    df_long.to_csv('WB_GDP_preprocessed_2_0.csv', index=False)


