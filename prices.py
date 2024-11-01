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

# Get stock data from the past year
aapl = yf.Ticker("AAPL")
hist = aapl.history("1y")

# Create column for timestamp with year:month:day
hist["timestamp"] = hist.index
hist['timestamp'] = hist['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d'))
print(hist.head())

# Retrieve news data from CSV and update timestamp to format year:month:day
news_data = pd.read_csv("data/AAPL.csv")
news_data['timestamp'] = news_data['timestamp'].apply(lambda x: x[:10])
print(news_data.head())

# Merge the news and stock data into one table
news_and_prices = pd.merge(hist, news_data, on='timestamp', how='inner')
print(news_and_prices.head())
print(news_and_prices.shape)

# Get the max price for every day in the past year
agg_prices = news_and_prices.groupby("timestamp").agg({"Open" : "max"})
print(agg_prices.head())
print(agg_prices.shape)

# Normalize the stock prices for the past year
agg_prices["Normalized"] = (agg_prices["Open"] - agg_prices["Open"].min()) / (agg_prices["Open"].max() - agg_prices["Open"].min())

# Plot normalized prices
# plt.plot(agg_prices["Open"])
plt.plot(agg_prices["Normalized"])
plt.xticks(ticks=range(0, len(agg_prices["Normalized"]), len(agg_prices["Normalized"]) // 11))
plt.xticks(rotation=45)
plt.show()