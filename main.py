from News import vaderpreprocess_text, news_fetch
from stock import Stock
import pandas as pd
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

def merge_by_month():
    df1 = pd.read_csv('historical_data.csv')
    df2 = pd.read_csv('stock_news.csv')
    df1['Date'] = pd.to_datetime(df1['Date'])
    df2['Date'] = pd.to_datetime(df2['Date'])
    df1['year_month'] = df1['date'].dt.to_period('M')
    df2['year_month'] = df2['date'].dt.to_period('M')
    merged_df = pd.merge(df1, df2, on='year_month', how='inner')
    merged_df = merged_df.drop(columns=['year_month'])
    merged_df.to_csv('merged_output.csv', index=False)

def merge_ai_csv():
    df1 = pd.read_csv('historical_data.csv')
    df2 = pd.read_csv('stock_news.csv')
    historical_data_cols = df1[['Date', 'Open_Shifted', 'Close', 'Close_Shifted']]
    technical_indicator_cols = df1[['RSI', 'SMA_50', 'EMA_20', 'MACD']]
    compound_sentiment = df2['Compound Sentiment']
    dataframe = pd.concat([historical_data_cols,technical_indicator_cols,compound_sentiment],axis=1)
    dataframe.to_csv('Compound_AI_csv.csv', index=False)

stock_dict = sp500_dict()
stock = input("Input a Stock or Stock Symbol: ")
stock_symbol, stock_name = check_user_stock(stock, stock_dict)
start_date = dt.datetime(2024, 1, 1)
end_date = dt.datetime(2024, 12, 31)

if stock_symbol is not None:
    news_fetch(stock_name)
    vaderpreprocess_text()
    example = Stock(stock_symbol, start_date, end_date)
    example.gather_data()
    example.generate_technical_signals()
    example.backtest()
    #example.plot_data()
    #merge_by_month()
    merge_ai_csv()


# example.add_technical_indicators()
