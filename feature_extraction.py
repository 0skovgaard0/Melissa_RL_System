import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.decomposition import PCA


class DataPreprocessor:
    def __init__(self, window_size):
        self.window_size = window_size

    def calculate_rolling_mean(self, data):
        return data.rolling(window=self.window_size).mean()

    def calculate_momentum(self, data):
        return data.pct_change(self.window_size)

    def calculate_volatility(self, data):
        return data.pct_change().rolling(window=self.window_size).std()

    def preprocess_data(self, data):
        data['Rolling_mean'] = data.groupby('Asset name')['Value'].transform(lambda x: self.calculate_rolling_mean(x))
        data['Momentum'] = data.groupby('Asset name')['Value'].transform(lambda x: self.calculate_momentum(x))
        data['Volatility'] = data.groupby('Asset name')['Value'].transform(lambda x: self.calculate_volatility(x))
        return data.dropna()

class DataPreparation:
    def __init__(self, lookback, n_features, test_size=0.2, pca_n_components=None):
        self.lookback = lookback
        self.n_features = n_features
        self.test_size = test_size
        self.pca_n_components = pca_n_components

    def apply_pca(self, X, n_components):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        print("PCA explained variance ratio:", pca.explained_variance_ratio_)
        return X_pca

    def create_lookback_data(self, data, y):
        num_samples = len(data) - self.lookback
        lookback_data = np.empty((num_samples, self.lookback, data.shape[1]))
        y = y[:num_samples]

        for i in range(num_samples):
            lookback_data[i] = data[i: i + self.lookback, :]

        return lookback_data, y

    def clean_data(self, data):
        return data


    def prepare_data(self, data, target_column):
       data = self.clean_data(data)

       X = data.drop([target_column, 'Date', 'Asset name', 'Series'], axis=1).values
       y = data[target_column].values

       # First, split the data into a (train + validation) set and a test set
       X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=False)

       # Then, split the (train + validation) set into a training set and a validation set
       X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=self.test_size, shuffle=False)

       if self.pca_n_components is not None:
           pca = PCA(n_components=self.pca_n_components)
           X_train_pca = pca.fit_transform(X_train)
           X_val_pca = pca.transform(X_val)
           X_test_pca = pca.transform(X_test)
           print("PCA explained variance ratio:", pca.explained_variance_ratio_)

           n_features_after_pca = self.pca_n_components if self.pca_n_components is not None else X_train.shape[1]
           X_train = X_train_pca
           X_val = X_val_pca
           X_test = X_test_pca

       X_train, y_train = self.create_lookback_data(X_train, y_train)
       X_val, y_val = self.create_lookback_data(X_val, y_val)
       X_test, y_test = self.create_lookback_data(X_test, y_test)

       return X_train, X_val, X_test, y_train, y_val, y_test



def test_preprocessing(data, expected_length):
    assert len(data) == expected_length, f"Data length mismatch: expected {expected_length}, got {len(data)}"


def main():
    # Read the data Finance data & Macro data
    financial_data = pd.read_csv("merged_financial_data.csv")
    macro_data = pd.read_csv("merged_macroeconomic_data.csv")

    # Filter to keep only the 'adjclose' series
    adjclose_data = financial_data[financial_data['Series'] == 'adjclose']

    if not os.path.exists('financial_data_features.csv'):
        # Preprocess data
        preprocessor = DataPreprocessor(window_size=5)
        adjclose_data = preprocessor.preprocess_data(adjclose_data.copy())
        macro_data = preprocessor.preprocess_data(macro_data.copy())

        adjclose_data.to_csv('financial_data_features.csv', index=False)
        macro_data.to_csv('macro_data_features.csv', index=False)
    else:
        adjclose_data = pd.read_csv('financial_data_features.csv')
        macro_data = pd.read_csv('macro_data_features.csv')

    # Test preprocessing
    test_preprocessing(adjclose_data, expected_length=249383) 

    print("Adjclose data shape:", adjclose_data.shape)
    print("Macro data shape:", macro_data.shape)

    # Combine the datasets
    combined_data = pd.concat([adjclose_data, macro_data], axis=0)

    print("\nCombined data shape:", combined_data.shape)
    print(combined_data.describe())
    print(f'\nCheck for missing values in the combined data\n', combined_data.isna().sum())


    # Save the combined data to a new CSV file if it doesn't exist
    if not os.path.exists('combined_data_features.csv'):
        combined_data.to_csv('combined_data_features.csv', index=False)
    else:
        combined_data = pd.read_csv('combined_data_features.csv')

    # Prepare the dataset for the 1D CNN model
    data_preparation = DataPreparation(lookback=10, n_features=2, test_size=0.2, pca_n_components=2)
    target_column = 'Value'  # target column name
    X_train, X_val, X_test, y_train, y_val, y_test = data_preparation.prepare_data(combined_data, target_column)

    print("\nX_train:", X_train[:5])
    print("X_val:", X_val[:5])
    print("X_test:", X_test[:5])
    print("y_train:", y_train[:5])
    print("y_val:", y_val[:5])
    print("y_test:", y_test[:5])
    print("\nX_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)
    print("y_test shape:", y_test.shape)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


    # We can now use X_train, X_test, y_train, and y_test to train and evaluate your 1D CNN model

if __name__ == "__main__":
    main()


