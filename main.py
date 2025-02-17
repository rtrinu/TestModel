from stock import Stock
import datetime as dt
import os


def main():
    stock = input("Input a Stock or Stock Symbol: ")
    start_date = dt.datetime(2024, 1, 1)
    end_date = dt.datetime(2024, 12, 31)
    example = Stock(stock, start_date, end_date)


if __name__ == "__main__":
    main()
