import csv
import requests
from bs4 import BeautifulSoup
import nltk
import pandas as pd

from nltk.sentiment import SentimentIntensityAnalyzer


#nltk.download('punkt')
#nltk.download('punkt_tab')
#nltk.download('averaged_perceptron_tagger_eng')
#nltk.download('maxent_ne_chunker_tab')
#nltk.download('words')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('vader_lexicon')


def fetch_rss_data(stock):
    url = requests.get(f'https://news.google.com/rss/search?q={stock}+stocks')
    soup = BeautifulSoup(url.content, 'xml')
    items = soup.find_all('item')

    with open("stock_news.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Link", "Published Date"])

    for item in items:
        title = item.title.text
        date = item.pubDate.text
        link = item.link.text
        with open("stock_news.csv", "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([title, link, date])


def Vaderpreprocess_text():
    df = pd.read_csv('stock_news.csv')
    if df.shape[0] > 100:
        df = df.head(20)
    sia = SentimentIntensityAnalyzer()
    res = []
    for i, row in df.iterrows():
        text = row["Title"]
        sentiment= sia.polarity_scores(text)
        res.append(sentiment['compound'])
    df['Compound Sentiment'] = res
    df.to_csv('stock_news.csv',index='false')
    print("Compound sentiment scores added to the CSV file under the 'Compound Sentiment' column.")



