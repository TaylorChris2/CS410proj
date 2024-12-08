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
from sentiment import sentiment_analysis, aggregation

# Get stock data from the past year
aapl = yf.Ticker("NVDA")
hist = aapl.history("1y")

# Create column for timestamp with year:month:day
hist["timestamp"] = hist.index
hist['timestamp'] = hist['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d'))
# print(hist.head())


# Retrieve news data from CSV and update timestamp to format year:month:day
news_data = pd.read_csv("data/NVDA_sentiment.csv")                        #Khushal  : Made a change here to use {stock_name}_sentiment.csv
# news_data['timestamp'] = news_data['timestamp'].apply(lambda x: x[:10])
news_data = sentiment_analysis(news_data)
news_data = aggregation(news_data)
news_data['timestamp'] = news_data['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d'))
# print(news_data.head())

# Merge the news and stock data into one table
news_and_prices = pd.merge(hist, news_data, on='timestamp', how='inner')
# print(news_and_prices.head())
# print(news_and_prices.shape)

# stock = 'NVDA'
# directory_name = 'data'
# filename = f"{stock}_final.csv"
# file_path = os.path.join(directory_name, filename)
# news_and_prices.to_csv(file_path)

# Get the max price for every day in the past year
# news_and_prices = news_and_prices.groupby("timestamp").agg({"Open" : "max"})
# print(news_and_prices.head())
# print(news_and_prices.shape)

# Normalize the stock prices for the past year
# news_and_prices["Normalized_prices"] = (news_and_prices["Open"] - news_and_prices["Open"].min()) / (news_and_prices["Open"].max() - news_and_prices["Open"].min())
# news_and_prices["Normalized_scores"] = (news_and_prices["sentiment_scores"] - news_and_prices["sentiment_scores"].min()) / (news_and_prices["sentiment_scores"].max() - news_and_prices["sentiment_scores"].min())

# # # Plot normalized prices
# # # plt.plot(news_and_prices["Open"])
# plt.plot(news_and_prices["Normalized_prices"])
# plt.plot(news_and_prices["Normalized_scores"])
# plt.xticks(ticks=range(0, len(news_and_prices["Normalized_prices"]), len(news_and_prices["Normalized_prices"]) // 11))
# plt.xticks(rotation=45)
# plt.show()

# # Pychart
# counts = {
#     'Neutral': (news_and_prices['sentiment_scores'] == 0).sum(),
#     'Positive': (news_and_prices['sentiment_scores'] > 0).sum(),
#     'Negative': (news_and_prices['sentiment_scores'] < 0).sum()
# }
# plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=140)
# plt.title('Perception')
# plt.show()