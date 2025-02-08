
from stock import Stock
from AiModels import run_models
import pandas as pd
import datetime as dt
import os


def main():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    stock = input("Input a Stock or Stock Symbol: ")
    start_date = dt.datetime(2024, 1, 1)
    end_date = dt.datetime(2024, 12, 31)
    example = Stock(stock, start_date, end_date)
    # example.plot_data()
    #run_models('Compound_AI.csv')


if __name__ == "__main__":
    main()
