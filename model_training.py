import pandas as pd
import tensorflow as tf
import numpy as np
from feature_extraction import DataPreprocessor, DataPreparation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

np.random.seed(0)
tf.random.set_seed(0)


class ModelTrainer:
    def __init__(self, input_shape, epochs, batch_size):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size

    def create_cnn_model(self):
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))

        # Add gradient clipping to the optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)

        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        return model

    def train_and_evaluate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test):
        history = model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, verbose=1)
        loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss: {loss}")
        return history

if __name__ == "__main__":
    # Instantiate the DataPreprocessor and DataPreparation classes
    preprocessor = DataPreprocessor(window_size=5)
    data_preparation = DataPreparation(lookback=10, n_features=11)

    # Read the updated_combined_data_features.csv
    combined_data = pd.read_csv('combined_data_features.csv')

    # Prepare the dataset for the 1D CNN model
    target_column = 'Value'  # target column name
    X_train, X_val, X_test, y_train, y_val, y_test = data_preparation.prepare_data(combined_data, target_column)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model_trainer = ModelTrainer(input_shape, epochs=100, batch_size=32)
    model = model_trainer.create_cnn_model()
    history = model_trainer.train_and_evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)

    print(f"CNN model input shape: {model.input_shape}")
    print(f"CNN model output shape: {model.output_shape}")
    
    # Save the CNN model
    model_path = 'C:/Users/janni/OneDrive/Skrivebord/VS Code/cnn_model.h5'
    model.save(model_path)
    print(f"Model saved at {model_path}")
