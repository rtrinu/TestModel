import pandas as pd

def fetch_sp500_data() -> dict:
        """
        Fetches the S&P 500 companies' information and returns a dictionary mapping
        stock symbols to company names.

        :return: dict - Dictionary with stock symbols as keys and company names as values.
        """
        url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
        sp500 = pd.read_csv(url)
        sp500.to_csv('sp500_stocks.csv', index=False)
        return dict(zip(sp500['Symbol'], sp500['Security']))

def get_stock_symbol_from_name(user_stock: str, stock_dict: dict) -> tuple:
        """
        Fetches the stock symbol from the stock dictionary using either the stock symbol or company name.

        :param user_stock: str, stock symbol or company name.
        :param stock_dict: dict, dictionary of stock symbols and company names.
        :return: tuple - Stock symbol and stock name (or None if not found).
        """
        if user_stock in stock_dict:
            return user_stock, stock_dict[user_stock]
        else:
            for symbol, name in stock_dict.items():
                if user_stock.lower() in name.lower():
                    return symbol, name
        return None, None