import os.path
from dictionary import fetch_sp500_data, get_stock_symbol_from_name
from News import StockNews
from lstmModel import lstmModel
from randomForestModel import randomForestModel
from SARHMM import SARHMM
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
        self.stock_dict = fetch_sp500_data()
        self.initialise()



    def gather_data(self, user_stock: str):
        """
        Gathers historical stock data for the given stock symbol or name.

        :param user_stock: str, stock symbol or company name to fetch data for.
        :return: None
        """
        stock_dict = fetch_sp500_data()
        user_stock_upper = user_stock.upper()
        stock_symbol, stock_name = get_stock_symbol_from_name(user_stock_upper, stock_dict)
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
        self.save_to_csv(f"{self.stock_name}_historical_data.csv")



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
        self.save_to_csv(f"{self.stock_name}_historical_data.csv")

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

        self.save_to_csv(f"{self.stock_name}_historical_data.csv")



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

    def save_to_csv(self, filename: str):
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

    def get_news_articles(self):
        """
        Gathers news articles about the company and creates a compound sentiment using outside methods from News
        :param user_stock: str, stock symbol or company name to fetch data for
        :return: None
        """
        news = StockNews(self.stock_name)

    def merge_ai_csv(self):
        """
        Merges historical stock data with news sentiment data and saves it to a new CSV file.

        This method reads in two CSV files: one containing historical stock data
        and another containing news sentiment data, and merges them into a single
        DataFrame which is then saved to a new CSV file.
        :return: None
        """
        try:
            data = pd.read_csv(f"{self.stock_name}_Compound_AI.csv")
            print("Compound CSV file already made")
        except FileNotFoundError:
            print("Creating CSV file")
            df1 = pd.read_csv(f"{self.stock_name}_historical_data.csv")
            df2 = pd.read_csv(f"{self.stock_name}_stock_news.csv")

            # Select relevant columns from the historical stock data
            historical_data_cols = df1[['Date', 'Open', 'Close', 'High', 'Low', 'Previous_Close', 'Volume']]
            technical_indicator_cols = df1[
                ['RSI', 'SMA_50', 'EMA_20', 'MACD', 'MACD_signal', 'MACD_histogram', 'Signal']]

            # Select the Compound Sentiment from the news data
            compound_sentiment = df2['Compound Sentiment']
            dataframe = pd.concat([historical_data_cols, technical_indicator_cols, compound_sentiment], axis=1)
            self.df = dataframe
            # Save the merged DataFrame to a new CSV file
            dataframe.to_csv(f'{self.stock_name}_Compound_AI.csv', index=False)

        # Load the historical stock data and news data



        #print(f"Merged data saved to '{self.stock_name}_Compound_AI.csv'.")

    def remove_csv_files(self):
        file_path = [f"{self.stock_name}_historical_data.csv", f"{self.stock_name}_stock_news.csv",
                     f"{self.stock_name}_Compound_AI.csv"]
        for file in file_path:
            if os.path.exists(file):
                os.remove(file)
                print(f"{file} deleted successfully")
            else:
                print(f"{file} not found")


    def train_ai_models(self):
        #lstm = lstmModel(f'{self.stock_name}_Compound_AI.csv', self.stock_name)
        rdfst = randomForestModel(f'{self.stock_name}_Compound_AI.csv')
        #hmm = SARHMM(f'{self.stock_name}_Compound_AI.csv')

        self.remove_csv_files()

    def initialise(self):
        """
        Initializes the process by gathering data, generating technical signals, and backtesting.

        :return: None
        """
        
       
        self.gather_data(self.stock_symbol)
        if self.stock_name == None:
            print("Invalid User Input")
        else:
            self.generate_technical_signals()
            self.get_news_articles()
            self.merge_ai_csv()

            
