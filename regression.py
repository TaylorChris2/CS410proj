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
# from sentiment import sentiment_analysis, aggregation

# stock = 'NVDA'
# directory_name = 'data'
# filename = f"{stock}_final.csv"
# file_path = os.path.join(directory_name, filename)

# df = pd.read_csv(file_path)


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
def pre_processing1(df, columns):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['quarter'] = df['timestamp'].apply(is_quarter_end)
    df['sentiment_scores'] = df['sentiment_scores'].apply(temp_func)
    df = df[columns]
    df = df.dropna()
    return df


# regression_columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'sentiment_scores', 'quarter']
# target = ['Close']

# df_lr = pre_processing(df, regression_columns)

# print("pre_processed data")

# features = ['timestamp','Open', 'High', 'Low','sentiment_scores', 'sentiment_score_lag','Close_Lag','quarter']
# target = ['Close']
# X = df_lr[features]
# y = df_lr[target]

def split_data(X, y):    
    split_value = int(len(X) * 0.8)
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


# print("split data")
# X_train, X_test, y_train, y_test, train_dates, test_dates = split_data(X, y)

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
    print(df.shape)
    return df, mae_test, mae_train, rmse_test, rmse_train

# predictions_lr, mae_test, mae_train, rmse_test, rmse_train = linear_regression(X_train, X_test, y_train, y_test)



# stock = 'NVDA'
# directory_name = 'data'
# filename = f"{stock}_predictions.csv"
# file_path = os.path.join(directory_name, filename)

# full_dates = train_dates.tolist() + test_dates.tolist()
# predictions_lr['Date'] = full_dates
# predictions_lr.set_index('Date', inplace=True)
# predictions_lr.to_csv(file_path)


##### All code for LSTM from here on
# scaler = StandardScaler()
# df_lstm = pre_processing1(df, regression_columns)

# features_lstm = ['timestamp','Open', 'High', 'Low','sentiment_scores','quarter']
# target_lstm = ['Close']
# numerical_cols = ['Open', 'High', 'Low']
# X_lstm = df_lstm[features_lstm]
# y_lstm = df_lstm[target_lstm]
# X_lstm.drop('timestamp', axis = 1, inplace = True)
# X_lstm[numerical_cols] = scaler.fit_transform(X_lstm[numerical_cols])
# y_lstm = scaler.fit_transform(y_lstm)

def create_sequences(X, y, n_steps):
    X_seq = []
    y_seq = []

    for i in range(n_steps, len(X)):
        X_seq.append(X.iloc[i - n_steps : i].values)
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# X_sequences, y_sequences = create_sequences(X_lstm, y_lstm, n_steps=3)

# print("X sequences shape:", X_sequences.shape) 
# print("y sequences shape:", y_sequences.shape)


# X_train, X_test, y_train, y_test = lstm_split(X_sequences, y_sequences)


# print("X train sequences shape:", X_train.shape) 
# print("y train sequences shape:", y_train.shape)

#Code adapted from https://www.analyticsvidhya.com/blog/2021/12/stock-price-prediction-using-lstm/
def LSTM_regression(X_train, y_train, X_test, y_test, scaler):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.1)) 
    model.add(Dense(units = 1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.fit(X_train, y_train, epochs=100,validation_data=(X_test, y_test), verbose=1)
    test_predicted = model.predict(X_test)
    train_predicted = model.predict(X_train)
    # scaled_test_predictions = Ms.inverse_transform(np.hstack([test_predicted, np.zeros((test_predicted.shape[0], 3))]))[:,0]
    # scaled_train_predictions = Ms.inverse_transform(np.hstack([train_predicted, np.zeros((train_predicted.shape[0], 3))]))[:,0]
    scaled_test_predictions = scaler.inverse_transform(test_predicted)
    scaled_train_predictions = scaler.inverse_transform(train_predicted)
    scaled_actual_y_train = scaler.inverse_transform(y_train)
    scaled_actual_y_test = scaler.inverse_transform(y_test)
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

# predictions_lstm, mae_test_lstm, mae_train_lstm, rmse_test_lstm, rmse_train_lstm = LSTM_regression(X_train, y_train, X_test, y_test)




# filename_lstm = f"{stock}_predictions_lstm.csv"
# file_path_lstm = os.path.join(directory_name, filename_lstm)

# predictions_lstm['Date'] = np.array(df['timestamp'])[3:]
# predictions_lstm.set_index('Date', inplace=True)
# predictions_lstm.to_csv(file_path_lstm)
    






