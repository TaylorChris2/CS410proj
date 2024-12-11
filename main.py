from dotenv import load_dotenv
from urllib.request import urlopen
from urllib.request import Request
from datetime import datetime, timedelta
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from urllib.request import urlopen
from urllib.request import Request
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import  Dense, Dropout, LSTM, Bidirectional # type: ignore
from sentiment import sentiment_analysis, aggregation
from eda import news 
from plots import create_dfs, regression_plot, regression_plot_predictions, lstm_plot, lstm_plot_predictions, both_plot_predictions
from regression import pre_processing, pre_processing1, split_data, lstm_split, linear_regression, LSTM_regression, create_sequences

directory_name = 'data'

os.makedirs(directory_name, exist_ok=True)
stock = 'INTC'
filename = f"{stock}.csv"
file_path = os.path.join(directory_name, filename)

def get_news(stock):
    df = news(stock)
    return df
headlines = get_news(stock)
headlines.to_csv(file_path)

headlines = pd.read_csv(file_path) #Read it again since we did some column name changing


def get_sentiment(df):
    df = sentiment_analysis(df)
    final_df = aggregation(df)
    return final_df

print(headlines.columns)
sentiment_df = get_sentiment(headlines)
filename = f"{stock}_sentiment.csv"

print("Got sentiment dataset")
file_path = os.path.join(directory_name, filename)


sentiment_df.to_csv(file_path)

stock_prices = yf.Ticker(stock)
hist = stock_prices.history("1y")

hist["timestamp"] = hist.index
hist['timestamp'] = hist['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d'))

news_data = pd.read_csv(f"data/{stock}_sentiment.csv")                        #Khushal  : Made a change here to use {stock_name}_sentiment.csv
# news_data['timestamp'] = news_data['timestamp'].apply(lambda x: x[:10])
# news_data['timestamp'] = news_data['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d'))

news_and_prices = pd.merge(hist, news_data, on='timestamp', how='inner')

filename = f"{stock}_final.csv"
file_path = os.path.join(directory_name, filename)

news_and_prices.to_csv(file_path)

print("Got training dataset")


df = pd.read_csv(file_path)

regression_columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'sentiment_scores', 'quarter']
target = ['Close']

df_lr = pre_processing(df, regression_columns)

features = ['timestamp','Open', 'High', 'Low','sentiment_scores', 'sentiment_score_lag','Close_Lag','quarter']
target = ['Close']
X = df_lr[features]
y = df_lr[target]

X_train, X_test, y_train, y_test, train_dates, test_dates = split_data(X, y)

predictions_lr, mae_test, mae_train, rmse_test, rmse_train = linear_regression(X_train, X_test, y_train, y_test)

filename = f"{stock}_predictions.csv"
file_path = os.path.join(directory_name, filename)

full_dates = train_dates.tolist() + test_dates.tolist()
predictions_lr['Date'] = full_dates
predictions_lr.set_index('Date', inplace=True)
predictions_lr.to_csv(file_path)

scaler = StandardScaler()
df_lstm = pre_processing1(df, regression_columns)

features_lstm = ['timestamp','Open', 'High', 'Low','sentiment_scores','quarter']
target_lstm = ['Close']
numerical_cols = ['Open', 'High', 'Low']
X_lstm = df_lstm[features_lstm]
y_lstm = df_lstm[target_lstm]
X_lstm.drop('timestamp', axis = 1, inplace = True)
X_lstm[numerical_cols] = scaler.fit_transform(X_lstm[numerical_cols])
y_lstm = scaler.fit_transform(y_lstm)

X_sequences, y_sequences = create_sequences(X_lstm, y_lstm, n_steps=3)

X_train, X_test, y_train, y_test = lstm_split(X_sequences, y_sequences)

predictions_lstm, mae_test_lstm, mae_train_lstm, rmse_test_lstm, rmse_train_lstm = LSTM_regression(X_train, y_train, X_test, y_test, scaler)

filename_lstm = f"{stock}_predictions_lstm.csv"
file_path_lstm = os.path.join(directory_name, filename_lstm)

predictions_lstm['Date'] = np.array(df['timestamp'])[3:]
predictions_lstm.set_index('Date', inplace=True)
predictions_lstm.to_csv(file_path_lstm)

print("---------regession results-------- \n\n")

print("Linear Regression Testing MAE:", round(mae_test, 5), '\n')
print("Linear Regression Training MAE:", round(mae_train, 5), '\n')

print("Linear Regression Testing RMSE:", round(rmse_test, 5), '\n')
print("Linear Regression Training RMSE:", round(rmse_train, 5), '\n')

print("---------LSTM regession results-------- \n\n")

print("LSTM Testing MAE:", round(mae_test_lstm, 5), '\n')
print("LSTM Training MAE:", round(mae_train_lstm, 5), '\n')

print("LSTM Testing RMSE:", round(rmse_test_lstm, 5), '\n')
print("LSTM Training RMSE:", round(rmse_train_lstm, 5), '\n')


regression_plot(stock)
lstm_plot(stock)

regression_plot_predictions(stock)
lstm_plot_predictions(stock)

both_plot_predictions(stock)
