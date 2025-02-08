from News import vaderpreprocess_text, news_fetch
from lstmModel import lstmModel
from randomForestModel import randomForestModel
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt


class Stock:
    def __init__(self, stock_symbol: str, start_date: str = None, end_date: str = None):
        """
        Initialize the Stock object with stock symbol and date range.

        :param stock_symbol: The symbol of the stock to analyze (e.g., 'AAPL').
        :param start_date: The start date for the stock data (optional).
        :param end_date: The end date for the stock data (optional).
        """
        self.stock_name = None
        self.stock_symbol = stock_symbol
        self.stock_symbol_upper = stock_symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        self.stock_dict = self.fetch_sp500_data()
        self.initialise()

    def fetch_sp500_data(self) -> dict:
        """
        Fetches the S&P 500 companies' information and returns a dictionary mapping
        stock symbols to company names.

        :return: dict - Dictionary with stock symbols as keys and company names as values.
        """
        url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
        sp500 = pd.read_csv(url)
        sp500.to_csv('sp500_stocks.csv', index=False)
        return dict(zip(sp500['Symbol'], sp500['Security']))

    def gather_data(self, user_stock: str):
        """
        Gathers historical stock data for the given stock symbol or name.

        :param user_stock: str, stock symbol or company name to fetch data for.
        :return: None
        """
        stock_dict = self.fetch_sp500_data()
        user_stock_upper = user_stock.upper()
        stock_symbol, stock_name = self.get_stock_symbol_from_name(user_stock_upper, stock_dict)
        self.stock_name = stock_name

        if not stock_symbol:
            print(f"Stock symbol or name '{user_stock}' not found in S&P 500.")
            return None

        print(f"Fetching data for '{stock_name}' ({stock_symbol})...")

        self.df = yf.download(stock_symbol, start=self.start_date, end=self.end_date)
        if self.df.empty:
            print(f"No data found for {stock_symbol}.")
            return None

        self.df = self.df.reset_index()
        self.add_technical_indicators()
        self.save_to_csv(f"{self.stock_symbol_upper}_historical_data.csv")

    def get_stock_symbol_from_name(self, user_stock: str, stock_dict: dict) -> tuple:
        """
        Fetches the stock symbol from the stock dictionary using either the stock symbol or company name.

        :param user_stock: str, stock symbol or company name.
        :param stock_dict: dict, dictionary of stock symbols and company names.
        :return: tuple - Stock symbol and stock name (or None if not found).
        """
        if user_stock in stock_dict:
            return user_stock, stock_dict[user_stock]
        else:
            for symbol, name in stock_dict.items():
                if user_stock.lower() in name.lower():
                    return symbol, name
        return None, None

    def add_technical_indicators(self):
        """
        Adds common technical indicators (RSI, SMA, EMA, MACD) to the stock DataFrame.

        :return: None
        """
        if self.df is None:
            print("No data found. Please gather data first.")
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
        self.save_to_csv(f"{self.stock_symbol_upper}_historical_data.csv")

    def generate_technical_signals(self):
        """
        Generates buy and sell signals based on technical indicators (EMA, SMA, MACD).

        :return: None
        """
        if self.df is None or self.df.empty:
            print("No data available to generate signals.")
            return

        self.df['Signal'] = 0
        buy_condition = (self.df['EMA_20'] > self.df['SMA_50']) & (self.df['MACD'] > self.df['MACD_signal'])
        sell_condition = (self.df['EMA_20'] < self.df['SMA_50']) & (self.df['MACD'] < self.df['MACD_signal'])

        self.df.loc[buy_condition, 'Signal'] = 1  # Buy signal
        self.df.loc[sell_condition, 'Signal'] = -1  # Sell signal

        self.save_to_csv(f"{self.stock_symbol_upper}_historical_data.csv")

    def backtest(self, initial_balance: int = 10000):
        """
        Backtests the trading strategy using buy and sell signals.

        :param initial_balance: int - The initial cash balance for the backtest.
        :return: float - The final balance after performing the backtest.
        """
        if self.df is None or self.df.empty:
            print("No data available for backtesting.")
            return 0.0

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
        Plots stock data, including closing prices, technical indicators (EMA, SMA),
        and signals (buy/sell) with RSI.

        :return: None
        """
        data = self.df
        fig, ax1 = plt.subplots(figsize=(14, 7), nrows=2, sharex=True)

        # Plot the stock prices and technical indicators on the first axis (ax1)
        ax1[0].plot(data['Close'], label='Close Price', alpha=0.5)
        ax1[0].plot(data['EMA_20'], label='EMA_20', linestyle='--', alpha=0.7)
        ax1[0].plot(data['SMA_50'], label='SMA_50', linestyle='--', alpha=0.7)

        # Plot Buy/Sell Signals on the price chart
        buy_signals = data[data['Signal'] == 1]
        sell_signals = data[data['Signal'] == -1]
        ax1[0].scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green')
        ax1[0].scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red')

        ax1[0].set_title('Stock Price with Indicators and RSI')
        ax1[0].set_ylabel('Price')
        ax1[0].legend(loc='upper left')

        # Plot RSI on the second axis (ax2)
        ax1[1].plot(data['RSI'], label='RSI', color='orange', alpha=0.7)
        ax1[1].axhline(30, color='blue', linestyle='--', alpha=0.7)  # Oversold line
        ax1[1].axhline(70, color='red', linestyle='--', alpha=0.7)  # Overbought line
        ax1[1].set_ylabel('RSI')
        ax1[1].set_xlabel('Date')
        ax1[1].legend(loc='upper left')

        plt.tight_layout()
        plt.show()

    def save_to_csv(self, filename:str):
        """
        Saves the stock data (contained in `self.df`) to a CSV file.

        :param filename: str - The name of the CSV file (default is 'historical_data.csv').
        :return: None
        """
        if self.df is not None:
            self.df.to_csv(filename, index=True)
            print(f"Data saved to {filename}")
        else:
            print("No data to save.")

    def get_news_articles(self, user_stock: str):
        """
        Gathers news articles about the company and creates a compound sentiment using outside methods from News
        :param user_stock: str, stock symbol or company name to fetch data for
        :return: None
        """
        stock_symbol, stock_name = self.get_stock_symbol_from_name(self.stock_symbol_upper, self.stock_dict)
        news_fetch(stock_name)
        vaderpreprocess_text()

    def merge_ai_csv(self):
        """
        Merges historical stock data with news sentiment data and saves it to a new CSV file.

        This method reads in two CSV files: one containing historical stock data
        and another containing news sentiment data, and merges them into a single
        DataFrame which is then saved to a new CSV file.
        :return: None
        """
        # Load the historical stock data and news data
        df1 = pd.read_csv("historical_data.csv")
        df2 = pd.read_csv("stock_news.csv")

        # Select relevant columns from the historical stock data
        historical_data_cols = df1[['Date', 'Open', 'Close','High', 'Low', 'Previous_Close','Volume']]
        technical_indicator_cols = df1[['RSI', 'SMA_50', 'EMA_20', 'MACD','MACD_signal', 'MACD_histogram', 'Signal']]

        # Select the Compound Sentiment from the news data
        compound_sentiment = df2['Compound Sentiment']

        # Merge all data into a single DataFrame
        dataframe = pd.concat([historical_data_cols, technical_indicator_cols, compound_sentiment], axis=1)

        # Save the merged DataFrame to a new CSV file
        dataframe.to_csv(f'{self.stock_symbol_upper}_Compound_AI.csv', index=False)
        #lstm = lstmModel(f'{self.stock_symbol_upper}_Compound_AI.csv')
        rdfst = randomForestModel(f'{self.stock_symbol_upper}_Compound_AI.csv')
        print("Merged data saved to 'Compound_AI.csv'.")

    def initialise(self):
        """
        Initializes the process by gathering data, generating technical signals, and backtesting.

        :return: None
        """
        self.gather_data(self.stock_symbol)
        self.generate_technical_signals()
        #self.backtest()
        self.get_stock_symbol_from_name(self.stock_symbol, self.stock_dict)
        self.merge_ai_csv()
