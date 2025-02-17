import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras.optimizers import Adam
import pickle


class lstmModel:
    def __init__(self, filename: str, stock_name: str, time_steps: int = 10, test_size: int = 0.2):
        self.robust_scaler = None
        self.minmax_scaler = None
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.model = None
        self.filename = filename
        self.stock_name = stock_name
        self.minmax_features = [
            'Close', 'Open', 'High', 'Low', 'Previous_Close',
            'SMA_50', 'EMA_20', 'MACD', 'MACD_signal', 'MACD_histogram'
        ]
        self.robust_features = ['RSI', 'Volume', 'Compound Sentiment']
        self.features = self.minmax_features + self.robust_features
        self.time_steps = time_steps
        self.test_size = test_size
        self.initialise()

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.filename)
        df = df.drop(columns=['Date', 'Price'], errors='ignore').dropna()

        minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        robust_scaler = RobustScaler()

        df[self.minmax_features] = minmax_scaler.fit_transform(df[self.minmax_features])
        df[self.robust_features] = robust_scaler.fit_transform(df[self.robust_features])

        x = df[self.features].values
        y = to_categorical(df['Signal'], num_classes=3)

        X_seq, y_seq = [], []
        for i in range(len(x) - self.time_steps):
            X_seq.append(x[i:i + self.time_steps])
            y_seq.append(y[i + self.time_steps])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(np.array(X_seq), np.array(y_seq),
                                                                                test_size=self.test_size,
                                                                                random_state=42)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def build_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units=100, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=3, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train_model(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=32):
        x_train = x_train.reshape(x_train.shape[0], self.time_steps, x_train.shape[2])
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(x_test, y_test),
                       callbacks=[early_stop]
                       )

    def evaluate_model(self, x_test, y_test):
        if x_test is None:
            print("x_test is None. Ensure data is loaded and preprocessed correctly.")
            return None, None
        x_test = x_test.reshape(x_test.shape[0], self.time_steps, x_test.shape[2])
        loss, accuracy = self.model.evaluate(x_test, y_test)
        return loss, accuracy

    def predict(self, x_input):
        x_input = x_input.reshape(x_input.shape[0], self.time_steps, x_input.shape[2])
        predictions = self.model.predict(x_input)
        predicted_classes = (predictions > 0.5).astype(int)
        print(predicted_classes)
        # print(classification_report(self.y_test, predicted_classes))

    def save_model(self):
        with open(f"{self.stock_name}_lstm_pickle", 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self):
        try:
            with open(f"{self.stock_name}_lstm_pickle", 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Model file not found. Proceeding to train a new model.")
            self.model = None
        except Exception as e:
            print(f"An unexpected error occurred while loading the model: {e}")

    def initialise(self):
        self.load_model()
        self.load_and_preprocess_data()
        if self.model is None:
            print("Training new model...")
            self.build_model(input_shape=(self.x_train.shape[1], self.x_train.shape[2]))
            self.train_model(self.x_train, self.y_train, self.x_test, self.y_test)
            loss, accuracy = self.evaluate_model(self.x_test, self.y_test)
            print(np.unique(self.y_train))  # Should output [0, 1]

            # predictions = self.predict(self.x_test)
            print(f"LSTM Model: \nTest Loss: {loss}, Test Accuracy: {accuracy}")
            self.save_model()
        else:
            print("Using saved model...")
            loss, accuracy = self.evaluate_model(self.x_test, self.y_test)
            print(f"LSTM Model: \nTest Loss: {loss}, Test Accuracy: {accuracy}")