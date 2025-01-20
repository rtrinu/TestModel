import csv

import pytz
import requests
from bs4 import BeautifulSoup
import nltk
from Keys import NewsAPI_Key
import pandas as pd
from datetime import datetime as dt, timedelta
from nltk.sentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import json


# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')

newsapi = NewsApiClient(NewsAPI_Key)
def fetch_rss_data(stock):
    url = requests.get(f'https://news.google.com/rss/search?q={stock}+stocks')
    soup = BeautifulSoup(url.content, 'xml')
    items = soup.find_all('item')

    news = []
    for item in items:
        title = item.title.text
        date = item.pubDate.text

    with open("stock_news.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Published Date"])

        for title, link, date in news:
            writer.writerow([title, date.strftime("%a, %d %b %Y %H:%M:%S %Z")])


def news_fetch(symbol):
    end_date = dt.today()
    start_date = end_date - timedelta(days=30)
    newsapi_response = newsapi.get_everything(
        q=symbol,  # Keyword or stock symbol to search for
        from_param=start_date,  # Start date (one month ago)
        to=end_date,  # End date (today's date)
        language='en',  # Language of the articles
        sort_by='publishedAt',  # Sort articles by publication date
        page_size=5  # Maximum number of results per page
    )

    # Process the articles returned from NewsAPI
    news_data = []
    if newsapi_response.get('articles'):
        for article in newsapi_response['articles']:
            title = article['title']
            published_at = article['publishedAt']
            news_data.append({
                'Title': title,
                'Published At': published_at
            })

    # Fetch articles using Google News RSS feed
    google_news_url = f'https://news.google.com/rss/search?q={symbol}+stocks'
    google_news_response = requests.get(google_news_url)
    soup = BeautifulSoup(google_news_response.content, 'xml')
    items = soup.find_all('item')

    for item in items:
        title = item.title.text
        published_at = item.pubDate.text
        news_data.append({
            'Title': title,
            'Published At': published_at
        })

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(news_data)

    # If the file doesn't exist, write the header; if it does exist, append the data
    df.to_csv('stock_news.csv', mode='a', header=not pd.io.common.file_exists('stock_news_csv'), index=False, quoting=1)

    # print(r.json)


def Vaderpreprocess_text():
    df = pd.read_csv('stock_news.csv')
    if df.shape[0] > 100:
        df = df.head(20)
    sia = SentimentIntensityAnalyzer()
    res = []
    for i, row in df.iterrows():
        text = row["Title"]
        sentiment = sia.polarity_scores(text)
        res.append(sentiment['compound'])
    df['Compound Sentiment'] = res
    df.to_csv('stock_news.csv', index='false')
    print("Compound sentiment scores added to the CSV file under the 'Compound Sentiment' column.")
