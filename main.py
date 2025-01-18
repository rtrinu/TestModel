from News import fetch_rss_data, Vaderpreprocess_text
from stock import Stock
import pandas as pd
import feedparser
import csv


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
print(stock_name, stock_symbol)
start_date = "2024-01-01"
end_date = "2024-12-31"
fetch_rss_data(stock_name)
Vaderpreprocess_text()
example = Stock(stock_symbol, start_date, end_date)
example.gather_data()
#example.add_technical_indicators()
