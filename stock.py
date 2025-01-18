import yfinance as yf
import pandas as pd
import pandas_ta as ta


class Stock:
    def __init__(self, stock_symbol, start_date, end_date):
        self.stock_symbol = stock_symbol
        self.start_date = start_date
        self.end_date = end_date
        self.df = None

    def gather_data(self):
        self.df = yf.download("MSFT", start=self.start_date, end=self.end_date)
        self.df = pd.DataFrame(self.df)
        self.df.columns = self.df.columns.get_level_values(0)
        self.df.dropna()
        self.df.reset_index()
        self.add_technical_indicators()
        self.save_to_csv('historical_data.csv')

    def add_technical_indicators(self):
        if self.df is None:
            print("No data found. Please gather data first")
            return

        self.df['RSI'] = ta.rsi(self.df['Close'], length=14)
        self.df['SMA_50'] = ta.sma(self.df['Close'], length=50)
        self.df['EMA_20'] = ta.ema(self.df['Close'], length=20)
        macd = ta.macd(self.df['Close'], length=14)
        print(macd.head())
        self.df['MACD'] = macd['MACD_12_26_9']
        self.df['MACD_signal'] = macd['MACDs_12_26_9']
        self.df['MACD_histogram'] = macd['MACDh_12_26_9']
        self.df = self.df.fillna(method='ffill',inplace=False)
        self.save_to_csv('historical_data.csv')



    def save_to_csv(self, filename):
        if self.df is not None:
            self.df.to_csv(filename, index=True)
            print(f"Data saved to {filename}")
        else:
            print("No data to save.")
