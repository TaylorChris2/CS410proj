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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, Dropout, LSTM, Bidirectional
# from sentiment import sentiment_analysis, aggregation

stock = 'INTC'
directory_name = 'data'
filename = f"{stock}_final.csv"
file_path = os.path.join(directory_name, filename)

df = pd.read_csv(file_path)

def temp_func(score):
    if score < 0:
        return -1
    elif (score > 0):
        return 1
    else:
        return 0


def is_quarter_end(date):
    if date.month in [3, 6, 9, 12]: 
        return 1
    else:
        return 0
def pre_processing(df, columns):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['quarter'] = df['timestamp'].apply(is_quarter_end)
    df['sentiment_scores'] = df['sentiment_scores'].apply(temp_func)
    df = df[columns]
    df['Close_Lag'] = df['Close'].shift(1)
    df['sentiment_score_lag'] = df['sentiment_scores'].shift(1)
    df = df.dropna()
    return df

regression_columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'sentiment_scores', 'quarter']
target = ['']

df = pre_processing(df, regression_columns)

print("pre_processed data")

features = ['timestamp','Open', 'High', 'Low','sentiment_scores', 'sentiment_score_lag','Close_Lag','quarter']
target = ['Close']
X = df[features]
y = df[target]

def split_data(X, y):    
    split_index = split_value = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:split_value], y.iloc[:split_value]
    X_test, y_test = X.iloc[split_value:], y.iloc[split_value:]
    train_dates = X_train['timestamp'].astype(str)
    test_dates = X_test['timestamp'].astype(str)


    X_train.drop('timestamp', axis = 1, inplace = True)
    X_test.drop('timestamp', axis = 1, inplace = True)
    return X_train, X_test, y_train, y_test, train_dates, test_dates

def lstm_split(X_sequences, y_sequences):
    split_index = int(len(X_sequences) * 0.8)
    
    X_train = X_sequences[:split_index]
    X_test = X_sequences[split_index:]
    
    y_train = y_sequences[:split_index]
    y_test = y_sequences[split_index:]
    
    return X_train, X_test, y_train, y_test


print("split data")
X_train, X_test, y_train, y_test, train_dates, test_dates = split_data(X, y)

def linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    train_predictions = model.predict(X_train)

    mae_test =  mean_absolute_error(y_test, test_predictions)
    mae_train = mean_absolute_error(y_train, train_predictions)

    rmse_test =  root_mean_squared_error(y_test, test_predictions)
    rmse_train = root_mean_squared_error(y_train, train_predictions)    

    y_complete = pd.concat([y_train, y_test])['Close'].tolist()
    predictions_complete = np.concatenate((train_predictions, test_predictions)).tolist()
    predictions_complete =  [pred[0] if isinstance(pred, list) else pred for pred in predictions_complete]
    tuples = list(zip(y_complete, predictions_complete))
    df = pd.DataFrame(tuples,
                    columns=['actual', 'predicted'])
    return df, mae_test, mae_train, rmse_test, rmse_train

predictions_lr, mae_test, mae_train, rmse_test, rmse_train = linear_regression(X_train, X_test, y_train, y_test)

print("---------regession results-------- \n\n")

print("Linear Regression Testing MAE:", round(mae_test, 5), '\n')
print("Linear Regression Training MAE:", round(mae_train, 5), '\n')

print("Linear Regression Testing RMSE:", round(rmse_test, 5), '\n')
print("Linear Regression Training RMSE:", round(rmse_train, 5), '\n')


stock = 'INTC'
directory_name = 'data'
filename = f"{stock}_predictions.csv"
file_path = os.path.join(directory_name, filename)

full_dates = train_dates.tolist() + test_dates.tolist()
predictions_lr['Date'] = full_dates
predictions_lr.set_index('Date', inplace=True)
predictions_lr.to_csv(file_path)


##### All code for LSTM from here on
Ms = MinMaxScaler()
numerical_cols = ['Open', 'High', 'Low', 'Close_Lag']
X_lstm = X.copy()
y_lstm = y.copy()
X_lstm[numerical_cols] = Ms.fit_transform(X_lstm[numerical_cols])
y_lstm = Ms.fit_transform(y_lstm)
X_lstm.drop('timestamp', axis = 1, inplace = True)

def create_sequences(X, y, n_steps):
    X_seq = []
    y_seq = []

    for i in range(n_steps, len(X)):
        X_seq.append(X.iloc[i - n_steps : i].values)
        y_seq.append(y.iloc[i])
    return np.array(X_seq), np.array(y_seq)

X_sequences, y_sequences = create_sequences(X_lstm, y, n_steps=3)

X_train, X_test, y_train, y_test = lstm_split(X_sequences, y_sequences)


print("X train sequences shape:", X_train.shape) 
print("y train sequences shape:", y_train.shape)

#Code adapted from https://www.analyticsvidhya.com/blog/2021/12/stock-price-prediction-using-lstm/
def LSTM_regression(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.1)) 
    model.add(Dense(units = 1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, epochs=30,validation_data=(X_test, y_test), verbose=1)
    test_predicted = model.predict(X_test)
    train_predicted = model.predict(X_train)
    scaled_test_predictions = Ms.inverse_transform(np.hstack([test_predicted, np.zeros((test_predicted.shape[0], 3))]))[:,0]
    scaled_train_predictions = Ms.inverse_transform(np.hstack([train_predicted, np.zeros((train_predicted.shape[0], 3))]))[:,0]
    scaled_actual_y_train = Ms.inverse_transform(y_train)
    scaled_actual_y_test = Ms.inverse_transform(y_test)
    mae_test =  mean_absolute_error(scaled_actual_y_test, scaled_test_predictions)
    mae_train = mean_absolute_error(scaled_actual_y_train, scaled_train_predictions)

    rmse_test =  root_mean_squared_error(scaled_actual_y_test, scaled_test_predictions)
    rmse_train = root_mean_squared_error(scaled_actual_y_train, scaled_train_predictions)

    y_complete = np.concatenate((scaled_actual_y_train, scaled_actual_y_test)).tolist()
    predictions_complete = np.concatenate((scaled_train_predictions, scaled_test_predictions)).tolist()
    tuples = list(zip(y_complete, predictions_complete))
    df = pd.DataFrame(tuples,
                    columns=['actual', 'predicted'])
    return df, mae_test, mae_train, rmse_test, rmse_train

predictions_lstm, mae_test_lstm, mae_train_lstm, rmse_test_lstm, rmse_train_lstm = LSTM_regression(X_train, y_train, X_test, y_test)

print("---------LSTM regession results-------- \n\n")

print("LSTM Testing MAE:", round(mae_test_lstm, 5), '\n')
print("LSTM Training MAE:", round(mae_train_lstm, 5), '\n')

print("LSTM Testing RMSE:", round(rmse_test_lstm, 5), '\n')
print("LSTM Training RMSE:", round(rmse_train_lstm, 5), '\n')




filename_lstm = f"{stock}_predictions_lstm.csv"
file_path_lstm = os.path.join(directory_name, filename_lstm)
predictions_lstm.to_csv(file_path_lstm)
    






