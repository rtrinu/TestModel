import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam


class lstmModel:
    def __init__(self, filename: str, time_steps: int = 10, test_size: int = 0.1):
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.model = None
        self.filename = filename
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
        y = df['Signal'].values

        X_seq, y_seq = [], []
        for i in range(len(x) - self.time_steps):
            X_seq.append(x[i:i + self.time_steps])
            y_seq.append(y[i + self.time_steps])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(np.array(X_seq), np.array(y_seq),
                                                                                test_size=self.test_size,
                                                                                random_state=42)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def build_model(self, input_shape):
        model = model = Sequential([
            Input(shape=input_shape),
            LSTM(units=100, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def train_model(self, x_train, y_train, x_test, y_test, epochs=50, batch_size=32):
        x_train = x_train.reshape(x_train.shape[0], self.time_steps, x_train.shape[2])
        early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        self.model.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(x_test, y_test),
                       callbacks=[early_stop]
                       )

    def evaluate_model(self,x_test,y_test):
        x_test=x_test.reshape(x_test.shape[0], self.time_steps, x_test.shape[2])
        loss,accuracy = self.model.evaluate(x_test,y_test)
        return loss, accuracy

    def predict(self,x_input):
        x_input = x_input.reshape(x_input.shape[0], self.time_steps, x_input.shape[2])
        predictions=self.model.predict(x_input)
        return predictions

    def initialise(self):
        self.load_and_preprocess_data()
        self.build_model(input_shape=(self.x_train.shape[1], self.x_train.shape[2]))
        self.train_model(self.x_train, self.y_train,self.x_test,self.y_test)
        loss, accuracy = self.evaluate_model(self.x_test, self.y_test)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")



