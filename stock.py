import yfinance as yf
import pandas as pd
import pandas_ta as ta
import datetime as dt
import matplotlib.pyplot as plt


class Stock:
    def __init__(self, stock_symbol, start_date=None, end_date=None):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        self.initialise()

    def sp500_dict(self):
        """
            Fetches the S&P 500 companies' information from an online CSV file,
            saves it to a local CSV file, and returns a dictionary mapping
            stock symbols to company names.

            :return: dict
                A dictionary with stock symbols as keys and company names as values.
            """
        url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
        sp500 = pd.read_csv(url)
        sp500.to_csv('sp500_stocks.csv', index=False)
        dictionary = dict(zip(sp500['Symbol'], sp500['Security']))
        return dictionary

    # def ftst_dict():

    def gather_data(self, user_stock: str):
        """
            Gathers historical stock data for the given stock symbol or name.

            :param user_stock: str, the stock symbol or name to fetch data for.
            :return: None, returns None if no data is found or invalid input.
            """
        stock_dict = self.sp500_dict()
        user_stock_upper = user_stock.upper()
        stock_symbol, stock_name = None, None
        if user_stock_upper in stock_dict:
            stock_symbol, stock_name = user_stock_upper, stock_dict[user_stock_upper]
        else:
            for symbol, name in stock_dict.items():
                if user_stock.lower() in name.lower():
                    stock_symbol, stock_name = symbol, name
                    break

        if not stock_symbol:
            print(f"Stock symbol or name '{user_stock}' not found in SP500.")
            return None

        print(f"Fetching data for '{stock_name}' ({stock_symbol})...")

        self.df = yf.download(self.stock_symbol, start=self.start_date, end=self.end_date)
        if self.df.empty:
            print(f"No data found for {stock_symbol}.")
            return None
        self.df = pd.DataFrame(self.df)
        self.df.columns = self.df.columns.get_level_values(0)
        self.df = self.df.dropna().reset_index()
        self.add_technical_indicators()
        self.save_to_csv('historical_data.csv')
        # self.RSI_plot()

    def add_technical_indicators(self):
        """
        This function adds common technical analysis indicators to the stock DataFrame.

        It first creates shifted versions of the stock's 'Open', 'High', 'Low', and 'Close' prices,
        then calculates various technical indicators including RSI, SMA, EMA, and MACD using the shifted closing prices.
        The indicators are used to assess the stock's historical performance and potential future trends.
        The function also removes rows with missing values and saves the updated DataFrame to a CSV file.

        :param self: object
            The instance of the class which holds the stock DataFrame and performs data analysis.

        :return: None
            The function directly modifies the DataFrame by adding the calculated indicators.
        """
        if self.df is None:
            print("No data found. Please gather data first")
            return

        self.df['Previous_Close'] = self.df['Close'].shift(1)
        self.df['Close_Shifted'] = self.df['Close'].shift(1)
        self.df['Open_Shifted'] = self.df['Open'].shift(1)
        self.df['High_Shifted'] = self.df['High'].shift(1)
        self.df['Low_Shifted'] = self.df['Low'].shift(1)

        self.df['RSI'] = ta.rsi(self.df['Close_Shifted'], length=14)
        self.df['SMA_50'] = ta.sma(self.df['Close_Shifted'], length=50)
        self.df['EMA_20'] = ta.ema(self.df['Close_Shifted'], length=20)
        macd = ta.macd(self.df['Close_Shifted'], length=14)
        self.df['MACD'] = macd['MACD_12_26_9']
        self.df['MACD_signal'] = macd['MACDs_12_26_9']
        self.df['MACD_histogram'] = macd['MACDh_12_26_9']
        self.df.dropna(inplace=True)
        self.save_to_csv('historical_data.csv')

    def RSI_plot(self):
        data = self.df
        combined = pd.DataFrame()
        combined['Close'] = data['Close']
        combined['RSI'] = data['RSI']

        plt.figure(figsize=(12, 8))
        ax1 = plt.subplot(211)
        ax1.plot(combined.index, combined['Close'], color='lightgray')
        ax1.set_title('Close Price', color='white')
        ax1.grid(True, color='#555555')
        ax1.set_axisbelow(True)
        ax1.set_facecolor('black')
        ax1.figure.set_facecolor('#121212')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')

        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(combined.index, combined['RSI'], color='lightgray')
        ax2.axhline(0, linestyle='--', alpha=0.5, color='#ff0000')
        ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
        ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
        ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
        ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
        ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
        ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
        ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')

        ax1.set_title('RSI Value', color='white')
        ax2.grid(False)
        ax2.set_axisbelow(True)
        ax2.set_facecolor('black')
        ax2.figure.set_facecolor('#121212')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')

        plt.show()

    def generate_technical_signals(self):
        """
        This function generates buy and sell signals based on technical analysis indicators.

        It checks if the DataFrame is valid and non-empty. The function then defines conditions for buy and sell signals
        based on the relationship between the Exponential Moving Average (EMA), Simple Moving Average (SMA), and the MACD indicator.
        If the EMA is greater than the SMA and the MACD is greater than the MACD signal, a buy signal is generated (denoted by 1).
        If the EMA is less than the SMA and the MACD is less than the MACD signal, a sell signal is generated (denoted by -1).
        The generated signals are added to the DataFrame, and the updated DataFrame is saved to a CSV file.

        :param self: object
            The instance of the class that holds the stock DataFrame with technical indicators.

        :return: None
            The function modifies the DataFrame in place by adding a 'Signal' column with generated buy/sell signals.
        """
        if self.df is None or self.df.empty:
            print("No data available to generate signals.")
            return

        self.df['Signal'] = 0
        buy_condition = (self.df['EMA_20'] > self.df['SMA_50']) & (self.df['MACD'] > self.df['MACD_signal'])
        sell_condition = (self.df['EMA_20'] < self.df['SMA_50']) & (self.df['MACD'] < self.df['MACD_signal'])

        self.df.loc[buy_condition, 'Signal'] = 1  # Buy signal
        self.df.loc[sell_condition, 'Signal'] = -1  # Sell signal

        self.save_to_csv('historical_data_with_signals.csv')

    def backtest(self, initial_balance: int = 10000):
        """
        This function backtests a trading strategy using buy and sell signals.

        It checks if the DataFrame containing stock data is valid and non-empty. The function then simulates a trading strategy by iterating over each row of the DataFrame.
        When a buy signal (1) is encountered, it simulates buying the stock by converting the available balance into a position at the current stock price.
        When a sell signal (-1) is encountered, it sells the position and converts it back to cash.
        After processing all the signals, the final balance is calculated by adding any remaining position's value at the current stock price.

        :param self: object
            The instance of the class that holds the stock data and buy/sell signals.

        :param initial_balance: int, default=10000
            The initial cash balance for the backtest. It represents the amount of money available to start the simulation.

        :return: float
            The final balance after executing the buy and sell signals.
        """
        if self.df is None or self.df.empty:
            print("No data available for backtesting.")
            return

        balance = initial_balance
        position = 0
        for _, row in self.df.iterrows():
            if row['Signal'] == 1:  # Buy
                position = balance / row['Close']
                balance = 0
            elif row['Signal'] == -1 and position > 0:  # Sell
                balance = position * row['Close']
                position = 0

        final_value = balance + (position * self.df['Close'].iloc[-1] if position > 0 else 0)
        print(f"Initial Balance: ${initial_balance}, Final Balance: ${final_value:.2f}")
        return final_value

    def plot_data(self):
        """
        This function plots stock data, including closing prices, technical indicators (EMA, SMA), and signals (buy/sell), along with the Relative Strength Index (RSI).

        The plot is divided into two subplots:
        1. The first subplot displays the stock's closing prices, the 20-day Exponential Moving Average (EMA), the 50-day Simple Moving Average (SMA), and the buy/sell signals.
           Buy signals are marked with green upward arrows (^) and sell signals with red downward arrows (v).
        2. The second subplot shows the RSI (Relative Strength Index) with horizontal lines indicating overbought (70) and oversold (30) levels.

        :param self: object
            The instance of the class that contains the stock data and technical indicators to be plotted.

        :return: None
            This function does not return any value; it only displays the plot.
        """
        data = self.df
        fig, ax1 = plt.subplots(figsize=(14, 7), nrows=2, sharex=True)

        # Plot the stock prices and technical indicators on the first axis (ax1)
        ax1[0].plot(data['Close'], label='Close Price', alpha=0.5)
        ax1[0].plot(data['EMA_20'], label='EMA_20', linestyle='--', alpha=0.7)
        ax1[0].plot(data['SMA_50'], label='SMA_50', linestyle='--', alpha=0.7)

        # Plot Buy/Sell Signals on the price chart
        buy_signals = data[data['Signal'] == -1]
        sell_signals = data[data['Signal'] == 1]
        ax1[0].scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green', alpha=1)
        ax1[0].scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red', alpha=1)

        # Title and labels for the stock price chart
        ax1[0].set_title('Stock Price with Indicators and RSI')
        ax1[0].set_ylabel('Price')
        ax1[0].legend(loc='upper left')

        # Plot RSI on the second axis (ax2)
        ax1[1].plot(data['RSI'], label='RSI', color='orange', alpha=0.7)
        ax1[1].axhline(y=30, color='blue', linestyle='--', alpha=0.7)  # Oversold line
        ax1[1].axhline(y=70, color='red', linestyle='--', alpha=0.7)  # Overbought line
        ax1[1].set_ylabel('RSI')
        ax1[1].set_xlabel('Date')
        ax1[1].legend(loc='upper left')

        # Show the plot
        plt.tight_layout()
        plt.show()

    def save_to_csv(self, filename:str):
        """
        This function saves the stock data (contained in the DataFrame `self.df`) to a CSV file.

        If the DataFrame is not empty, it writes the data to the specified CSV file with the given filename.
        The index of the DataFrame is included in the CSV file. A confirmation message is printed upon successful saving.
        If no data is available (i.e., `self.df` is None), a message indicating that no data is available to save is printed.

        :param self: object
            The instance of the class that contains the DataFrame to be saved.

        :param filename: str
            The name of the CSV file where the data will be saved.

        :return: None
            This function does not return any value, it only saves the data to a file.
        """
        if self.df is not None:
            self.df.to_csv(filename, index=True)
            print(f"Data saved to {filename}")
        else:
            print("No data to save.")

    def initialise(self):
        """
        This function initializes the entire process of gathering data, generating technical signals, and performing a backtest.

        It first calls `gather_data()` to fetch the stock data, followed by `generate_technical_signals()` to compute relevant technical indicators
        and trading signals (such as buy/sell conditions). Finally, it performs a backtest of the strategy using the `backtest()` function to simulate
        the performance of the trading signals over the historical data.

        :param self: object
            The instance of the class that contains methods for gathering data, generating signals, and backtesting.

        :return: None
            This function does not return any value, it merely orchestrates the data gathering, signal generation, and backtesting process.
        """

        self.gather_data()
        self.generate_technical_signals()
        self.backtest()
