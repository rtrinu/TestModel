import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from statsmodels.tsa.ar_model import AutoReg


class SARHMM():
    def __init__(self, filename: str, n_states=3, ar_lags=5):
        self.filename = filename
        self.n_states = n_states
        self.ar_lags = ar_lags
        self.features = None
        self.returns = None
        self.data = None
        self.hmm_model = None
        self.ar_models = {}
        self.initialise()

    def load_and_preprocess_data(self):
        """Load and preprocess stock data, calculate returns, and prepare features."""
        self.data = pd.read_csv(self.filename, parse_dates=['Date'], index_col='Date')

        cols_to_convert = ['Close', 'Previous_Close', 'RSI', 'SMA_50', 'EMA_20',
                           'MACD', 'MACD_signal', 'MACD_histogram', 'Signal', 'Compound Sentiment']
        for col in cols_to_convert:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        self.data.dropna(inplace=True)
        self.data['Returns'] = np.log(self.data['Close'] / self.data['Previous_Close'])
        self.features = self.data[['RSI', 'SMA_50', 'EMA_20', 'MACD', 'MACD_signal',
                                   'MACD_histogram', 'Signal', 'Compound Sentiment']].values
        self.returns = self.data['Returns'].values

    def train_hmm(self):
        """Train a Hidden Markov Model on the features to predict market regimes."""
        self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=1000)
        self.model.fit(self.features)
        self.data['State'] = self.model.predict(self.features)

    def fit_ar_model(self):
        """Fit autoregressive models for each detected market regime."""
        for state in range(self.n_states):
            state_data = self.data[self.data['State'] == state]['Returns']
            if len(state_data) > self.ar_lags:
                ar_model = AutoReg(state_data, lags=self.ar_lags).fit()
                self.ar_models[state] = ar_model

    def predict_next_regime(self):
        """Predict the next market regime using the trained Hidden Markov Model."""
        latest_features = self.features[-1].reshape(1, -1)
        predicted_state = self.model.predict(latest_features)[0]
        return predicted_state

    def predict_next_return(self):
        """Predict the next return based on the autoregressive model of the predicted regime."""
        predicted_state = self.predict_next_regime()
        if predicted_state in self.ar_models:
            past_returns = self.data['Returns'].iloc[-self.ar_lags:].values
            if len(past_returns) < self.ar_lags:
                print(f"Not enough past return data! Required: {self.ar_lags}, Available: {len(past_returns)}")
                return None

            predicted_return = self.ar_models[predicted_state].predict(start=len(self.data), end=len(self.data))
            predicted_return = predicted_return.iloc[-1]

            print(f"Predicted Return Output: {predicted_return}")
            if isinstance(predicted_return, np.ndarray) and len(predicted_return) > 0:
                print(f"Predicted Next Return: {predicted_return[0]:.6f}")
                return predicted_return[0]
            else:
                print("Error: Predicted return is empty.")
                return None
        else:
            print("No AR model available for predicted regime.")
            return None

    def backtest_strategy(self):
        """Simulate a simple trading strategy based on predicted market regimes."""
        self.data['Signal'] = 0
        self.data.loc[self.data['State'] == 0, 'Signal'] = 1  # Buy in bullish regime
        self.data.loc[self.data['State'] == 1, 'Signal'] = -1  # Short sell in bearish regime

        self.data['Strategy_Return'] = self.data['Signal'].shift(1) * self.data['Returns']
        self.data['Cumulative_Return'] = self.data['Strategy_Return'].cumsum()

        plt.figure(figsize=(12, 5))
        plt.plot(self.data.index, self.data['Cumulative_Return'], label="Strategy Return", color="blue")
        plt.plot(self.data.index, self.data['Returns'].cumsum(), label="Market Return", linestyle="dashed", color="red")
        plt.legend()
        plt.title("Trading Strategy vs Market Performance")
        plt.show()

    def initialise(self):
        """Initialise the model by loading data, training the HMM, fitting AR models, and running backtest."""
        self.load_and_preprocess_data()
        self.train_hmm()
        self.fit_ar_model()
        self.predict_next_return()
        self.backtest_strategy()
