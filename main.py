from News import fetch_rss_data, Vaderpreprocess_text
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import feedparser
import csv
from bs4 import BeautifulSoup
import requests

fetch_rss_data()
Vaderpreprocess_text()
df = yf.download("MSFT",start="2024-01-01",end="2024-12-31")
df = pd.DataFrame(df)