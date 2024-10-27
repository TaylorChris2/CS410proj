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
print("Test")

## iterate through the dataset, headline collum, get each headline that will be he sentance
#iterate through data set
#ss is array of score.
scores = []
for sentence in df["headline"]:
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(sentence)
    sorted_ss = sorted(ss)
     #add a collum
    for k in sorted(ss):
        scores.append(ss[k])
        break
    
df["sentiment_score"] = scores
print(df.head())
print(df["sentiment_score"])