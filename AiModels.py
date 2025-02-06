import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.src.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    classification_report
from sklearn.preprocessing import StandardScaler


def LSTM_model_creation(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    features = ['Close', 'Close_Shifted', 'RSI', 'SMA_50', 'EMA_20', 'Compound Sentiment', 'Open', 'Open_Shifted',
                'Signal']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    sequence_length = 60
    x = []
    y = []

    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i, 0])

    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.8)
    input_shape = (sequence_length, len(features))
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    LSTM_model_train(model, x_train, y_train, x_test, y_test, 75, 32)
    make_stock_prediction(model, df, features)
    lstm_plot(df, model, features)


def LSTM_model_train(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
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


def random_forest_creation(filename):
    # Load and preprocess the data
    df = pd.read_csv(filename)
    df = df.dropna()

    # Define features and target
    features = ['Close', 'Close_Shifted', 'RSI', 'SMA_50', 'EMA_20', 'Compound Sentiment', 'Open', 'Open_Shifted']
    target = 'Signal'
    sequence_length = 60

    # Prepare sequences
    x = []
    y = []

    for i in range(sequence_length, len(df)):
        x.append(df[features].iloc[i - sequence_length:i])
        y.append(df[target].iloc[i])

    x = np.array(x)
    y = np.array(y)

    # Flatten the sequences for the RandomForestClassifier
    x_flat = np.array([sequence[-1] for sequence in x])

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x_flat, y, shuffle=False, train_size=0.8)

    # Initialize and train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    random_forest_train(x_train, y_train, x_test, y_test, model)
    tree_plot(df, model, features)


def random_forest_train(x_train, y_train, x_test, y_test, model):
    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(x_test)

    # Convert y_test to pandas Series for alignment
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, name="Target")

    # Align predictions with y_test index
    predictions = pd.Series(predictions, index=y_test.index, name="Predictions")

    # Combine actual and predicted values
    combined = pd.concat([y_test, predictions], axis=1)
    combined.columns = ["Target", "Predictions"]

    # Calculate accuracy
    accuracy = (combined["Target"] == combined["Predictions"]).mean()
    print(f"Model Accuracy: {accuracy:.2f}")

    # Print confusion matrix and classification report for multiclass classification
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    return model




def make_stock_prediction(model, df, features, sequence_length=60):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    x_input = scaled_data[-sequence_length:].reshape((1, sequence_length, len(features)))
    predicted_scaled = model.predict(x_input)
    predicted_scaled_reshaped = np.zeros((1, len(features)))
    predicted_scaled_reshaped[0, 0] = predicted_scaled[0, 0]
    predicted_price = scaler.inverse_transform(predicted_scaled_reshaped)

    return predicted_price[0][0]


def lstm_plot(df, model, features, sequence_length=60):
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


def tree_plot(df, model, features, sequence_length=60):
    """
    Function to plot actual vs predicted signals (buy, sell, hold) compared to historical stock prices using Random Forest Classifier.

    Parameters:
    - df: DataFrame containing the stock data (must include price_column).
    - model: Trained Random Forest Classifier model.
    - features: List of feature columns to use for prediction.
    - sequence_length: The number of previous days to consider for prediction (default is 60).
    """

    # Prepare the data to predict the next day's signal
    x = []
    for i in range(sequence_length, len(df)):
        # Flatten the sequence of features into a 1D vector for each sample
        x.append(df[features].iloc[i - sequence_length:i].values.flatten())

    x = np.array(x)  # Now x will be 2D: (n_samples, sequence_length * features)

    # Predict the signals for the test set
    predictions = model.predict(x)

    # Get the historical prices for the last sequence_length days
    historical_prices = df['Close'].iloc[-sequence_length:]  # Last 'sequence_length' days

    # Get the actual next day's price (the day after the last data point)
    actual_next_day_price = df['Close'].iloc[-1]

    # Plot the historical prices and predicted signals
    plt.figure(figsize=(14, 7))
    plt.plot(df.index[-sequence_length:], historical_prices, label='Historical Prices', color='blue')

    # Get the indices of the predictions (after the historical sequence)
    predicted_dates = df.index[-len(predictions):]

    # Plot predicted signals: Buy, Sell, Hold (1, -1, 0)
    buy_signals = predicted_dates[predictions == 1]
    sell_signals = predicted_dates[predictions == -1]
    hold_signals = predicted_dates[predictions == 0]

    # Scatter plot for buy, sell, hold signals
    plt.scatter(buy_signals, df.loc[buy_signals, 'Close'], marker="^", color="green", label="Buy Signal", alpha=1)
    plt.scatter(sell_signals, df.loc[sell_signals, 'Close'], marker="v", color="red", label="Sell Signal", alpha=1)
    plt.scatter(hold_signals, df.loc[hold_signals, 'Close'], marker="o", color="orange", label="Hold Signal", alpha=1)

    # Add actual next day's price (for comparison)
    plt.plot(len(historical_prices), actual_next_day_price, 'go', label='Actual Price (Next Day)', markersize=8)

    # Customize plot
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Actual Stock Prices and Predicted Buy/Sell/Hold Signals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print actual vs predicted values
    print(f"Actual Next Day Price: {actual_next_day_price}")
    print(f"Predicted Signals for Next Day: {predictions[-1]}")
