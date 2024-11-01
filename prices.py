import pandas as pd
import numpy as np
import finnhub
import os
from dotenv import load_dotenv
from urllib.request import urlopen
from urllib.request import Request
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt

aapl = yf.Ticker("AAPL")
hist = aapl.history("1y")

hist["timestamp"] = hist.index
hist['timestamp'] = hist['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d'))
print(hist.head())

news_data = pd.read_csv("data/AAPL.csv")
news_data['timestamp'] = news_data['timestamp'].apply(lambda x: x[:10])
print(news_data.head())

news_and_prices = pd.merge(hist, news_data, on='timestamp', how='inner')
print(news_and_prices.head())
print(news_and_prices.shape)

agg_prices = news_and_prices.groupby("timestamp").agg({"Open" : "max"})
print(agg_prices.head())
print(agg_prices.shape)

# plt.plot(hist["Open"])
plt.plot(agg_prices["Open"])
plt.show()