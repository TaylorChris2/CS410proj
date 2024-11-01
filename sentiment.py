import pandas as pd
import numpy as np
import finnhub
import os
from dotenv import load_dotenv
from urllib.request import urlopen
from urllib.request import Request
from datetime import datetime, timedelta
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')


stock = 'AAPL'
directory_name = 'data'
filename = f"{stock}.csv"
file_path = os.path.join(directory_name, filename)
df = pd.read_csv(file_path)
print(df.head())




def sentiment_analysis(news):
    sid = SentimentIntensityAnalyzer()
    scores = pd.DataFrame(df['headline'].astype(str).apply(sid.polarity_scores).to_list())
    news['sentiment_scores'] = scores['compound']
    return news

def aggregation(df):
    df['timestamp'] = df['timestamp'].apply(lambda x : x[:10])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    final_news_df = df.groupby('timestamp', as_index = False)['sentiment_scores'].mean()
    final_news_df.sort_values(by='timestamp', inplace=True)
    return final_news_df

df = sentiment_analysis(df)
final_df = aggregation(df)

print(final_df.shape)
print(final_df.head())