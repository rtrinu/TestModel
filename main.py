from News import fetch_rss_data, Vaderpreprocess_text, requests, get_news
from stock import Stock
import pandas as pd
import feedparser
import csv
import datetime as dt


def sp500_dict():
    url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
    sp500 = pd.read_csv(url)
    sp500.to_csv('sp500_stocks.csv', index=False)
    sp500_dict = dict(zip(sp500['Symbol'], sp500['Security']))
    return sp500_dict

def check_user_stock(user_stock, stock_dict):
    stock_symbols = None
    stock_name = None
    if user_stock.upper() in stock_dict:
        stock_symbols = user_stock.upper()
        stock_name = stock_dict[stock_symbols]
        print(f"Stock symbol '{user_stock.upper()}' found: {stock_dict[user_stock.upper()]}")
        return stock_symbols, stock_name
    elif any(user_stock.lower() in name.lower() for symbol, name in stock_dict.items()):
        for symbol, name in stock_dict.items():
            if user_stock.lower() in name.lower():
                stock_symbols = symbol
                stock_name = name
                print(f"Stock name '{user_stock}' found: Symbol is {symbol}")
                return stock_symbols, stock_name
    else:
        print(f"Stock symbol or name '{user_stock}' not found in SP500.")
        return None, None


stock_dict = sp500_dict()
stock = input("Input a Stock or Stock Symbol: ")
stock_symbol, stock_name = check_user_stock(stock, stock_dict)
start_date = dt.datetime(2024,1,1)
end_date = dt.datetime(2024,12,31)

Vaderpreprocess_text()
if stock_symbol is not None:
    get_news(stock_symbol, start_date, end_date)
    example = Stock(stock_symbol, start_date, end_date)
    example.gather_data()
    df1 = pd.read_csv('historical_data.csv')
    historical_data_cols = df1[['Close','Open_Shifted','Close_Shifted']]
    technical_indicator_cols = df1[['RSI','SMA_50','EMA_20','MACD']]
    df1 = pd.read_csv('stock_news.csv')

#example.add_technical_indicators()
