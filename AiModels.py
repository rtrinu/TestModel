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


def LSTM_model_creation(filename: str):
    """
    This function creates, trains, and evaluates an LSTM model for stock price prediction based on historical data.

    It loads a dataset from the provided CSV file, processes the data (scaling and feature engineering), and prepares the
    data for training the LSTM model. The model is then trained on the training data and evaluated on the test data.
    Predictions for stock prices are generated, and the results are plotted.

    :param filename: str
        The path to the CSV file containing the stock data. The file should include columns such as 'Close', 'Close_Shifted',
        'RSI', 'SMA_50', 'EMA_20', 'Compound Sentiment', 'Open', 'Open_Shifted', and 'Signal'.

    :return: None
        This function does not return any value. It trains the LSTM model, makes predictions, and visualizes results.
    """
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


def LSTM_model_train(model: object, x_train: list, y_train: list, x_test: list, y_test: list, epochs: int,
                     batch_size: int):
    """
        This function trains an LSTM model using the provided training data and evaluates it on the test data.

        The model is trained using the specified number of epochs and batch size. It utilizes early stopping to monitor
        validation loss and stop training if the validation loss does not improve for a given number of epochs.
        After training, it plots the training and validation loss curves.

        :param model: object
            The LSTM model to be trained. This model should be a compiled Keras model.

        :param x_train: list
            The input features for training. It should be a 3D array of shape (samples, sequence_length, features).

        :param y_train: list
            The target values for training. It should be a 1D array of shape (samples,).

        :param x_test: list
            The input features for testing. It should have the same shape as x_train.

        :param y_test: list
            The target values for testing. It should have the same shape as y_train.

        :param epochs: int
            The number of epochs to train the model for.

        :param batch_size: int
            The number of samples per gradient update (batch size) during training.

        :return: None
            This function does not return any value. It trains the LSTM model and plots the training/validation loss curves.
        """
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
    """
        This function plots the training and validation loss curves for an LSTM model.

        The function takes the training history of an LSTM model and visualizes the loss values during training
        and validation over each epoch. This helps to analyze the model's performance and whether it is overfitting.

        :param history: object
            The training history object returned by the `fit` method of the Keras model. It contains the loss
            values for each epoch, which are accessed via `history.history['loss']` for training loss and
            `history.history['val_loss']` for validation loss.

        :return: None
            This function does not return any value. It simply plots the loss curves.
        """
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def random_forest_creation(filename: str):
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


def random_forest_train(x_train: list, y_train: list, x_test: list, y_test: list, model: object):
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


def make_stock_prediction(model: object, df: pd.DataFrame, features: list, sequence_length: int = 60):
    """
        This function uses a trained model to make a stock price prediction based on historical data.

        It scales the provided features using StandardScaler, then reshapes the data into the format required by the model
        (for time series prediction) and uses the model to make a prediction for the next time step.
        Finally, it inversely scales the predicted price back to the original scale of the data.

        :param model: object
            The trained machine learning model (e.g., LSTM or any other model) used to make the prediction.

        :param df: pd.DataFrame
            The DataFrame containing the historical stock data, including the features used for prediction.

        :param features: list
            A list of column names (strings) from the DataFrame representing the features used in making predictions.

        :param sequence_length: int, default=60
            The number of pastime steps to use for making the prediction. For example, if set to 60,
            the model will use the previous 60 days of data to predict the next day's stock price.

        :return: float
            The predicted stock price for the next time step (after the last row in the DataFrame).
        """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    x_input = scaled_data[-sequence_length:].reshape((1, sequence_length, len(features)))
    predicted_scaled = model.predict(x_input)
    predicted_scaled_reshaped = np.zeros((1, len(features)))
    predicted_scaled_reshaped[0, 0] = predicted_scaled[0, 0]
    predicted_price = scaler.inverse_transform(predicted_scaled_reshaped)

    return predicted_price[0][0]


def lstm_plot(df: pd.DataFrame, model: object, features: list, sequence_length: int = 60):
    """
        This function visualizes the historical stock prices along with the actual and predicted prices
        for the next day. It uses the trained model to make the prediction and compares it with the actual
        next day's stock price.

        It plots the historical stock prices, the actual next day's price, and the predicted next day's price
        on the same graph to help evaluate the model's performance.

        :param df: pd.DataFrame
            The DataFrame containing the stock data. It should include the 'Close' column (historical stock prices).

        :param model: object
            The trained machine learning model (e.g., LSTM or other model) used for making predictions.

        :param features: list
            A list of feature columns (strings) used to train the model for predictions. These features are necessary
            for making predictions.

        :param sequence_length: int, default=60
            The number of previous days of data used for prediction (i.e., the sequence length). The model will use
            the last `sequence_length` days to predict the next day's price.

        :return: None
            This function does not return any value, but it displays a plot showing the historical stock prices,
            actual price, and predicted price.
        """
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


def tree_plot(df: pd.DataFrame, model: object, features: list, sequence_length: int = 60):
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


def run_models(filename: str):
    LSTM_model_creation(filename)
