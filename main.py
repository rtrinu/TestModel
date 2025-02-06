from News import vaderpreprocess_text, news_fetch
from stock import Stock
from AiModels import run_models
import pandas as pd
import datetime as dt


def main():
    stock = input("Input a Stock or Stock Symbol: ")
    start_date = dt.datetime(2024, 1, 1)
    end_date = dt.datetime(2024, 12, 31)
    example = Stock(stock, start_date, end_date)
    # example.plot_data()
    run_models('Compound_AI.csv')


if __name__ == "__main__":
    main()
