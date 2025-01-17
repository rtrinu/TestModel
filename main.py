from News import fetch_rss_data, Vaderpreprocess_text
from stock import Stock
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import feedparser
import csv
from bs4 import BeautifulSoup
import requests

stock = input("Input a stock: ")
start_date = "2024-01-01"
end_date = "2024-12-31"
# fetch_rss_data(stock)
# Vaderpreprocess_text()
example = Stock(stock, start_date, end_date)
example.gather_data()
#example.add_technical_indicators()
