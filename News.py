import csv
import requests
from bs4 import BeautifulSoup
import nltk
from Keys import RapidAPI_Key
import pandas as pd
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
import json

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('vader_lexicon')


def fetch_rss_data(stock, start_date, end_date):
    url = requests.get(f'https://news.google.com/rss/search?q={stock}+stocks')
    soup = BeautifulSoup(url.content, 'xml')
    items = soup.find_all('item')

    news = []
    for item in items:
        title = item.title.text
        date = item.pubDate.text
        formatted_date = dt.strptime(date, "%a, %d %b %Y %H:%M:%S %Z")

        if start_date <= formatted_date <= end_date:
            news.append((title, formatted_date))

    news.sort(key=lambda x: x[2], reverse=True)

    with open("stock_news.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Link", "Published Date"])

        for title, link, date in news:
            writer.writerow([title, link, date.strftime("%a, %d %b %Y %H:%M:%S %Z")])


def get_news(symbol, start_date, end_date):
    url = "https://seeking-alpha.p.rapidapi.com/news/v2/list-by-symbol"
    querystring = {"id":symbol}
    headers = {
        "x-rapidapi-key": RapidAPI_Key,
        "x-rapidapi-host": "seeking-alpha.p.rapidapi.com"
    }

    news_titles = []
    news_dates = []
    df = pd.DataFrame()
    response = requests.get(url, headers=headers, params=querystring).json()
    for article in response['data']:
        article_title = article['attributes'].get('title','No title found')
        publish_date = article['attributes'].get('publishOn','No Date found')
        news_titles.append(article_title)
        news_dates.append(publish_date)
    df['Title'] = news_titles
    df['Date'] = news_dates
    df.to_csv('articles.csv',index=False)


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
