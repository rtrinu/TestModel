from stock import Stock
from dictionary import fetch_sp500_data, get_stock_symbol_from_name
import datetime as dt
import os


def main():
    dict = fetch_sp500_data()
    stock = input("Input a Stock or Stock Symbol: ")
    symbol =get_stock_symbol_from_name(stock,dict)
    while symbol == None:
        stock = input("Input a Stock or Stock Symbol: ")
    start_date = dt.datetime(2024, 1, 1)
    end_date = dt.datetime(2024, 12, 31)
    example = Stock(stock, start_date, end_date)
    


if __name__ == "__main__":
    main()
