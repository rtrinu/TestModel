import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def LSTM_model_creation(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    features = ['Close', 'Close_Shifted', 'RSI', 'SMA_50', 'EMA_20', 'Compound Sentiment', 'Open', 'Open_Shifted']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    sequence_length = 60
    x = []
    y = []

    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i,0])

    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.8)
    input_shape = (sequence_length, len(features))
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=100, return_sequences=True),  # First LSTM layer with return_sequences=True
        Dropout(0.2),  # Regularization
        LSTM(units=50, return_sequences=False),  # Second LSTM layer
        Dropout(0.2),  # Regularization
        Dense(units=1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    LSTM_model_train(model, x_train, y_train,x_test, y_test, 75, 32)
    make_stock_prediction(model,df, features)
    plot_comparison(df, model, features)
def LSTM_model_train(model, x_train, y_train,x_test, y_test, epochs, batch_size):
    early_stop = EarlyStopping(monitor='val_loss',patience=50, restore_best_weights=True)
    history = model.fit(
        x_train, y_train,
        epochs = epochs,
        batch_size = batch_size,
        validation_data = (x_test,y_test),
        callbacks=[early_stop],
        verbose=1
    )
    plot_lstm(history)

def plot_lstm(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def make_stock_prediction(model, df, features, sequence_length=60):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    x_input = scaled_data[-sequence_length:].reshape((1, sequence_length, len(features)))
    predicted_scaled = model.predict(x_input)
    predicted_scaled_reshaped = np.zeros((1, len(features)))
    predicted_scaled_reshaped[0, 0] = predicted_scaled[0, 0]
    predicted_price = scaler.inverse_transform(predicted_scaled_reshaped)

    return predicted_price[0][0]


def plot_comparison(df, model, features, sequence_length=60):
    # Predict stock price
    predicted_price = make_stock_prediction(model, df, features, sequence_length)

    # Get the actual historical prices
    historical_prices = df['Close'].iloc[-sequence_length:]  # Last sequence_length days

    # Get the actual next day's price
    actual_next_day_price = df['Close'].iloc[-1]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(historical_prices)), historical_prices, label='Historical Prices', color='blue')
    plt.plot(len(historical_prices), actual_next_day_price, 'go', label='Actual Price (Next Day)',
             markersize=8)  # Green marker
    plt.plot(len(historical_prices), predicted_price, 'ro', label='Predicted Price (Next Day)',
             markersize=8)  # Red marker
    plt.xlabel('Time (Days)')
    plt.ylabel('Stock Price')
    plt.title('Actual Historical Prices, Actual Next Day Price, and Predicted Price')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Actual Next Day Price: {actual_next_day_price}")
    print(f"Predicted Next Day Price: {predicted_price}")


# Visualize comparison
