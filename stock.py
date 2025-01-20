import yfinance as yf
import pandas as pd
import pandas_ta as ta
import datetime as dt
import matplotlib.pyplot as plt

class Stock:
    def __init__(self, stock_symbol, start_date, end_date):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.df = None

    def gather_data(self):
        self.df = yf.download(self.stock_symbol, start=self.start_date, end=self.end_date)
        self.df = pd.DataFrame(self.df)
        self.df.columns = self.df.columns.get_level_values(0)
        self.df.dropna()
        self.df.reset_index()
        self.add_technical_indicators()
        self.save_to_csv('historical_data.csv')
        # self.RSI_plot()

    def add_technical_indicators(self):
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

        plt.figure(figsize=(12,8))
        ax1=plt.subplot(211)
        ax1.plot(combined.index,combined['Close'],color='lightgray')
        ax1.set_title('Close Price',color='white')
        ax1.grid(True, color='#555555')
        ax1.set_axisbelow(True)
        ax1.set_facecolor('black')
        ax1.figure.set_facecolor('#121212')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y',colors = 'white')

        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(combined.index, combined['RSI'],color='lightgray')
        ax2.axhline(0,linestyle= '--',alpha=0.5,color='#ff0000')
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
        data = self.df
        data['Signal'] = 0
        for i in range(len(data)):
            if(
                #data['RSI'].iloc[i]<30 or
                data['EMA_20'].iloc[i] > data['SMA_50'].iloc[i] and
                data['MACD'].iloc[i] > data['MACD_signal'].iloc[i]
            ):
                data.at[data.index[i],'Signal'] = 1
            elif(
                    #data['RSI'].iloc[i] > 70 or
                    data['EMA_20'].iloc[i] < data['SMA_50'].iloc[i] and
                    data['MACD'].iloc[i] < data['MACD_signal'].iloc[i]
            ):
                data.at[data.index[i],'Signal'] = -1
        data.to_csv('historical_data.csv')

    def backtest(self, initial_balance=10000):
        data = self.df
        balance = initial_balance
        position = 0
        for i in range(len(data)):
            if data['Signal'].iloc[i] == 1:
                position = balance / data['Close'].iloc[i]
                balance = 0
            elif data['Signal'].iloc[i] == -1 and position > 0:
                balance = position * data['Close'].iloc[i]
                position = 0
        final_value = balance + (position * data['Close'].iloc[-1]if position > 0  else 0)
        return final_value

    def plot_data(self):
        data = self.df
        """Plot stock data with indicators, RSI, and signals."""

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
    def save_to_csv(self, filename):
        if self.df is not None:
            self.df.to_csv(filename, index=True)
            print(f"Data saved to {filename}")
        else:
            print("No data to save.")
