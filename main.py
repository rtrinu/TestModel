from News import vaderpreprocess_text, news_fetch
from stock import Stock
from AiModels import run_models
import pandas as pd
import datetime as dt





def merge_ai_csv():
    df1 = pd.read_csv('historical_data.csv')
    df2 = pd.read_csv('stock_news.csv')
    historical_data_cols = df1[['Date','Open', 'Open_Shifted', 'Close', 'Close_Shifted']]
    technical_indicator_cols = df1[['RSI', 'SMA_50', 'EMA_20', 'MACD','Signal']]
    compound_sentiment = df2['Compound Sentiment']
    dataframe = pd.concat([historical_data_cols,technical_indicator_cols,compound_sentiment],axis=1)
    dataframe.to_csv('Compound_AI.csv', index=False)


if "__name__" == "__main__":
    stock_dict = sp500_dict()
    stock = input("Input a Stock or Stock Symbol: ")
    stock_symbol, stock_name = check_user_stock(stock, stock_dict)
    start_date = dt.datetime(2024, 1, 1)
    end_date = dt.datetime(2024, 12, 31)

    if stock_symbol is not None:
        news_fetch(stock_name)
        vaderpreprocess_text()
        example = Stock(stock_symbol, start_date, end_date)
        # example.plot_data()
        # merge_by_month()
        merge_ai_csv()
        run_models('Compound_AI.csv')