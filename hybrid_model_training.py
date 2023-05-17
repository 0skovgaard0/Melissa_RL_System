from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Lambda
import pandas as pd
from xgboost import XGBRegressor
from feature_extraction import DataPreprocessor, DataPreparation
import os
import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

class HybridModelTrainer:
    def __init__(self, input_shape, epochs, batch_size):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size

    def load_cnn_model(self, model_path):
        if os.path.exists(model_path):
            model = load_model(model_path)
            return model
        else:
            raise Exception(f"No model found at {model_path}")

    def create_hybrid_model(self, xgb_model):
        # Wrap the XGBoost model with a Keras model
        input = Input(shape=self.input_shape)
        output = Lambda(lambda x: xgb_model.predict(x))(input)
        model = Model(inputs=input, outputs=output)
        return model

    def train_and_evaluate_model(self, cnn_model, X_train, X_val, X_test, y_train, y_val, y_test):
        # Create a new model that's the same as the CNN model up to the flatten layer
        feature_extractor = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-3].output)

        # Use the feature extractor to generate features from the raw time-series data
        X_train_features = feature_extractor.predict(X_train)
        X_val_features = feature_extractor.predict(X_val)
        X_test_features = feature_extractor.predict(X_test)

        # test feature shape
        print(X_train_features.shape)
        print(X_val_features.shape)
        print(X_test_features.shape)

        # Train the XGBoost model
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train_features, y_train, early_stopping_rounds=10, eval_set=[(X_val_features, y_val)], verbose=True)

        # Wrap the trained XGBoost model with a Keras model
        hybrid_model = self.create_hybrid_model(xgb_model)

        return hybrid_model


if __name__ == "__main__":
    # Instantiate the DataPreprocessor and DataPreparation classes
    preprocessor = DataPreprocessor(window_size=5)
    data_preparation = DataPreparation(lookback=10, n_features=11)

    # Read the combined_data_features.csv
    combined_data = pd.read_csv('combined_data_features.csv')

    # Prepare the dataset for the hybrid model
    target_column = 'Value'  # target column name
    X_train, X_val, X_test, y_train, y_val, y_test = data_preparation.prepare_data(combined_data, target_column)

    # Load the trained CNN model
    model_trainer = HybridModelTrainer(input_shape=(X_train.shape[1], X_train.shape[2]), epochs=100, batch_size=32)
    cnn_model = model_trainer.load_cnn_model('C:/Users/janni/OneDrive/Skrivebord/VS Code/cnn_model.h5')

    # Train and evaluate the hybrid model
    hybrid_model = model_trainer.train_and_evaluate_model(cnn_model, X_train, X_val, X_test, y_train, y_val, y_test)

    print(f"Hybrid model input shape: {hybrid_model.input_shape}")
    print(f"Hybrid model output shape: {hybrid_model.output_shape}")

    # Save the hybrid model
    hybrid_model_path = 'C:/Users/janni/OneDrive/Skrivebord/VS Code/hybrid_model.h5'
    hybrid_model.save(hybrid_model_path)
    print(f"Hybrid Model saved at {hybrid_model_path}")

