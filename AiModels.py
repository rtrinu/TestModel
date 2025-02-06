import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


def create_and_train_lstm_model(filename: str):
    """
    Loads stock data, processes it, trains an LSTM model, and visualizes results.

    :param filename: str - path to the stock data CSV file
    """
    df = load_and_preprocess_data(filename)

    features = ['Close', 'Close_Shifted', 'RSI', 'SMA_50', 'EMA_20', 'Compound Sentiment', 'Open', 'Open_Shifted',
                'Signal']
    x, y = prepare_lstm_sequences(df, features, sequence_length=60)

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.8)

    model = build_lstm_model(input_shape=(60, len(features)))
    train_lstm_model(model, x_train, y_train, x_test, y_test, epochs=75, batch_size=32)

    predict_and_visualize_lstm(df, model, features)


def load_and_preprocess_data(filename: str):
    """Load and clean the stock data from a CSV file."""
    df = pd.read_csv(filename)
    df = df.dropna()
    return df


def prepare_lstm_sequences(df: pd.DataFrame, features: list, sequence_length: int = 60):
    """
    Prepare sequences of features and labels for LSTM model.

    :param df: pd.DataFrame - DataFrame containing stock data
    :param features: list - list of feature columns
    :param sequence_length: int - number of past steps to consider for prediction
    :return: x, y - input features and corresponding labels
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])

    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i, 0])

    return np.array(x), np.array(y)


def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model.

    :param input_shape: tuple - shape of input data for LSTM
    :return: model - compiled LSTM model
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_lstm_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    """
    Train the LSTM model with early stopping.

    :param model: object - compiled LSTM model
    :param x_train: list - training data
    :param y_train: list - training labels
    :param x_test: list - testing data
    :param y_test: list - testing labels
    :param epochs: int - number of epochs for training
    :param batch_size: int - batch size for training
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                        callbacks=[early_stop], verbose=1)

    plot_training_history(history)


def plot_training_history(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def predict_and_visualize_lstm(df, model, features):
    """
    Use the trained LSTM model to make predictions and visualize the results.

    :param df: pd.DataFrame - historical stock data
    :param model: object - trained LSTM model
    :param features: list - list of features used for prediction
    """
    predicted_price = make_lstm_prediction(model, df, features)
    actual_price = df['Close'].iloc[-1]

    plot_predictions(df, predicted_price, actual_price)


def make_lstm_prediction(model, df, features, sequence_length=60):
    """Make a stock price prediction using the trained LSTM model."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    x_input = scaled_data[-sequence_length:].reshape((1, sequence_length, len(features)))
    predicted_scaled = model.predict(x_input)

    predicted_scaled_reshaped = np.zeros((1, len(features)))
    predicted_scaled_reshaped[0, 0] = predicted_scaled[0, 0]

    predicted_price = scaler.inverse_transform(predicted_scaled_reshaped)
    return predicted_price[0][0]


def plot_predictions(df, predicted_price, actual_price):
    """
    Plot historical prices, predicted price, and actual next day's price.

    :param df: pd.DataFrame - historical stock data
    :param predicted_price: float - predicted price for the next day
    :param actual_price: float - actual stock price for the next day
    """
    historical_prices = df['Close'].iloc[-60:]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(historical_prices)), historical_prices, label='Historical Prices', color='blue')
    plt.plot(len(historical_prices), actual_price, 'go', label='Actual Price', markersize=8)
    plt.plot(len(historical_prices), predicted_price, 'ro', label='Predicted Price', markersize=8)
    plt.xlabel('Time (Days)')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Prediction: Actual vs Predicted')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Actual Price: {actual_price}")
    print(f"Predicted Price: {predicted_price}")


def create_and_train_random_forest_model(filename: str):
    """
    Train a Random Forest model to predict buy/sell/hold signals based on stock data.

    :param filename: str - path to the stock data CSV file
    """
    df = load_and_preprocess_data(filename)

    features = ['Close', 'Close_Shifted', 'RSI', 'SMA_50', 'EMA_20', 'Compound Sentiment', 'Open', 'Open_Shifted']
    target = 'Signal'
    x, y = prepare_random_forest_data(df, features, target, sequence_length=60)

    model = train_random_forest_model(x, y)
    evaluate_random_forest_model(model, x, y)
    plot_random_forest_predictions(df, model, features)


def prepare_random_forest_data(df, features, target, sequence_length=60):
    """
    Prepare sequences for training a Random Forest model.

    :param df: pd.DataFrame - DataFrame containing stock data
    :param features: list - list of feature columns
    :param target: str - target column for classification
    :param sequence_length: int - number of past steps to consider for prediction
    :return: x, y - input features and target labels
    """
    x, y = [], []
    for i in range(sequence_length, len(df)):
        x.append(df[features].iloc[i - sequence_length:i].values.flatten())
        y.append(df[target].iloc[i])

    return np.array(x), np.array(y)


def train_random_forest_model(x, y):
    """
    Train a Random Forest Classifier.

    :param x: list - input features for training
    :param y: list - target labels for training
    :return: model - trained Random Forest model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    model.fit(x, y)
    return model


def evaluate_random_forest_model(model, x, y):
    predictions = model.predict(x)
    accuracy = accuracy_score(y, predictions)
    print(f"Random Forest Model Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, predictions))


def plot_random_forest_predictions(df, model, features):
    """Visualize the predicted buy/sell/hold signals from the Random Forest model."""
    x = []
    for i in range(60, len(df)):
        x.append(df[features].iloc[i - 60:i].values.flatten())

    predictions = model.predict(np.array(x))
    predicted_dates = df.index[-len(predictions):]

    plt.figure(figsize=(14, 7))
    plt.plot(df.index[-60:], df['Close'].iloc[-60:], label='Historical Prices', color='blue')
    plt.scatter(predicted_dates, df.loc[predicted_dates, 'Close'], marker="o", label="Predicted Signals")
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price and Predicted Buy/Sell/Hold Signals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_models(filename: str):
    """
    Run both LSTM and Random Forest models for stock price prediction and signal generation.

    :param filename: str - path to the stock data CSV file
    """
    create_and_train_lstm_model(filename)
    create_and_train_random_forest_model(filename)
