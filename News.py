import csv
import requests
from bs4 import BeautifulSoup
import nltk
import pandas as pd
from matplotlib import pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


def fetch_rss_data():
    keywords = ["Tesla", "Apple", "Amazon", "Google", "DraftKings","NASA"]
    stock = input("Input a stock: ")
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
        if any(keyword.lower() in title.lower() for keyword in keywords):
            with open("stock_news.csv", "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([title, link, date])


def Vaderpreprocess_text():
    df = pd.read_csv('stock_news.csv')
    if df.shape[0] > 100:
        df = df.head(20)

    example = df['Title'][20]
    sia = SentimentIntensityAnalyzer()
    print(example)
    print(sia.polarity_scores(example))

    res = {}
    for i, row in df.iterrows():
        text = row["Title"]
        res[text] = sia.polarity_scores(text)
    vaders = pd.DataFrame(res).T
    vaders = vaders.reset_index().rename(columns={'index': 'Id'})
    vaders = vaders.merge(df, how='left', left_on='Id', right_on='Title')
    sns.lineplot(data=vaders, x='Published Date', y='compound')
    plt.show()


