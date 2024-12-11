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
# stock = "NVDA"

def sentiment_plots(stock):
    # Retrieve news data from CSV
    stock_data = pd.read_csv(f"data/{stock}_final.csv")

    # Normalize the stock prices and sentiment scores for the past year
    stock_data["normalized_prices"] = (stock_data["Close"] - stock_data["Close"].min()) / (stock_data["Close"].max() - stock_data["Close"].min())
    stock_data["normalized_scores"] = (stock_data["sentiment_scores"] - stock_data["sentiment_scores"].min()) / (stock_data["sentiment_scores"].max() - stock_data["sentiment_scores"].min())

    # Plot normalized prices with normalized sentiment scores
    plt.plot(stock_data["normalized_prices"])
    plt.plot(stock_data["normalized_scores"])
    plt.title(f"Stock and sentiment for {stock}")
    plt.show()

    # Pie chart for sentiment scores
    counts = {
        'Neutral': (stock_data['sentiment_scores'] == 0).sum(),
        'Positive': (stock_data['sentiment_scores'] > 0).sum(),
        'Negative': (stock_data['sentiment_scores'] < 0).sum()
    }
    plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=140)
    plt.title(f'Perception of {stock}')
    plt.show()