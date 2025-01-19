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

    def technincal_indicator_training(self):
        window = 20
        indicators = ['RSI','EMA_20','SMA_50','MACD','Close_Shifted','Previous_Close']
        results={indicators: {'predictions': [],'actual':[],'daily_mae':[]} for indicator in indicators}

    def save_to_csv(self, filename):
        if self.df is not None:
            self.df.to_csv(filename, index=True)
            print(f"Data saved to {filename}")
        else:
            print("No data to save.")
