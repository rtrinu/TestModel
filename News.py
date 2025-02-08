import os
import csv
import pytz
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime as dt, timedelta
from dotenv import load_dotenv
from newsapi import NewsApiClient
from nltk.sentiment import SentimentIntensityAnalyzer


# Load environment variables
class StockNews:
    def __init__(self,symbol:str) -> None:
        """
        Initializes the StockNewsAnalyzer class.

        :param news_api_key: str - Your NewsAPI key.
        """
        self.configure()
        self.newsapi_key = os.getenv('NEWS_KEY')
        self.newsapi = NewsApiClient(api_key=self.newsapi_key)
        self.sia = SentimentIntensityAnalyzer()
        self.symbol = symbol

        self.initalise()

    def configure(self) -> None:
        """
        Configure the environment by loading environment variables from a .env file.
        """
        load_dotenv()

    def news_fetch(self, symbol: str) -> None:
        """
        Fetch the latest stock news articles for a given stock symbol from NewsAPI and Google News.

        :param symbol: str - The stock symbol (e.g., 'AAPL' for Apple)
        """
        end_date = dt.today()
        start_date = end_date - timedelta(days=30)

        # Fetch news from NewsAPI
        newsapi_response = self.newsapi.get_everything(
            q=symbol,
            from_param=start_date,
            to=end_date,
            language='en',
            sort_by='publishedAt',
            page_size=1
        )

        news_data = []

        # Parse the response from NewsAPI
        if newsapi_response.get('articles'):
            for article in newsapi_response['articles']:
                title = article['title']
                published_at = dt.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").strftime('%Y-%m-%d')
                news_data.append({
                    'Title': title,
                    'Date': published_at
                })

        # Fetch additional news from Google News
        google_news_url = f'https://news.google.com/rss/search?q={symbol}+stocks'
        google_news_response = requests.get(google_news_url)
        soup = BeautifulSoup(google_news_response.content, 'xml')
        items = soup.find_all('item')

        for item in items:
            title = item.title.text
            published_at = dt.strptime(item.pubDate.text, "%a, %d %b %Y %H:%M:%S %Z").strftime('%Y-%m-%d')
            news_data.append({
                'Title': title,
                'Date': published_at
            })

        # Sort news articles by date in descending order
        news_data_sorted = sorted(news_data, key=lambda x: x['Date'], reverse=True)
        df = pd.DataFrame(news_data_sorted)

        # Save the news data to a CSV file
        df.to_csv(f'{symbol}_stock_news.csv', index=False)
        print(f"News data for {symbol} saved to '{symbol}_stock_news.csv'.")

    def vaderpreprocess_text(self, csv_file: str) -> None:
        """
        Process the text data from the CSV file, analyze sentiment using VADER, and add a column for compound sentiment scores.

        :param csv_file: str - The name of the CSV file (e.g., 'AAPL_stock_news.csv').
        """
        df = pd.read_csv(csv_file)

        # List to store the sentiment scores
        res = []

        # Analyze sentiment for each news title
        for i, row in df.iterrows():
            text = row["Title"]
            sentiment = self.sia.polarity_scores(text)
            res.append(sentiment['compound'])

        # Add the sentiment scores to the DataFrame
        df['Compound Sentiment'] = res

        # Save the updated DataFrame back to CSV
        df.to_csv(csv_file, index=False)
        print(f"Compound sentiment scores added to the CSV file '{csv_file}' under the 'Compound Sentiment' column.")

    def initalise(self):
        self.news_fetch(self.symbol)
        self.vaderpreprocess_text(f"{self.symbol}_stock_news.csv")