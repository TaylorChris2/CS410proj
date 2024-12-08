import matplotlib.pyplot as plt
import pandas as pd
import os


def create_dfs(stock):
    # stock = 'INTC'
    directory_name = 'data'

    regression_filename = f"{stock}_predictions.csv"
    regression_file_path = os.path.join(directory_name, regression_filename)

    LSTM_filename = f"{stock}_predictions_lstm.csv"
    LSTM_file_path = os.path.join(directory_name, LSTM_filename)

    df_regression = pd.read_csv(regression_file_path)
    df_LSTM = pd.read_csv(LSTM_file_path)

    df_LSTM['actual'] = df_LSTM['actual'].apply(lambda x: float(x[1:len(x)-1]))
    df_LSTM['predicted'] = df_LSTM['predicted'].apply(lambda x: float(x[1:len(x)-1]))

    regression_actual = df_regression['actual'].tolist()
    regression_predicted = df_regression['predicted'].tolist()

    LSTM_actual = df_LSTM['actual'].tolist()
    LSTM_predicted = df_LSTM['predicted'].tolist()

    return regression_actual, regression_predicted, LSTM_actual, LSTM_predicted

def regression_plot(stock):
    regression_actual, regression_predicted, LSTM_actual, LSTM_predicted = create_dfs(stock)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(regression_actual, label='Actual Prices', color='blue')
    plt.plot(regression_predicted, label='Predicted Prices', color='orange')

    # Adding titles and labels
    plt.title('Stock Price Prediction Using Linear Regression')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()

    # Show plot
    plt.show()

def lstm_plot(stock):
    regression_actual, regression_predicted, LSTM_actual, LSTM_predicted = create_dfs(stock)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(LSTM_actual, label='Actual Prices', color='blue')
    plt.plot(LSTM_predicted, label='Predicted Prices', color='orange')

    # Adding titles and labels
    plt.title('Stock Price Prediction Using LSTM')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()

    # Show plot
    plt.show()

def regression_plot_predictions(stock):
    regression_actual, regression_predicted, LSTM_actual, LSTM_predicted = create_dfs(stock)

    regression_split_value = int(0.8 * len(regression_actual))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(regression_actual[:regression_split_value+1], label='Actual Train Prices', color='blue')
    plt.plot(range(regression_split_value, len(regression_actual)), regression_actual[regression_split_value:], label='Actual Test Prices', color='green')
    plt.plot(range(regression_split_value, len(regression_actual)), regression_predicted[regression_split_value:], label='Predicted Test Prices', color='red')

    # Adding titles and labels
    plt.title('Stock Price Prediction Using Linear Regression')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()

    # Show plot
    plt.show()

def lstm_plot_predictions(stock):
    regression_actual, regression_predicted, LSTM_actual, LSTM_predicted = create_dfs(stock)

    LSTM_split_value = int(0.8 * len(regression_actual))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(LSTM_actual[:LSTM_split_value+1], label='Actual Train Prices', color='blue')
    plt.plot(range(LSTM_split_value, len(LSTM_actual)), LSTM_actual[LSTM_split_value:], label='Actual Test Prices', color='green')
    plt.plot(range(LSTM_split_value, len(LSTM_actual)), LSTM_predicted[LSTM_split_value:], label='Predicted Test Prices', color='red')

    # Adding titles and labels
    plt.title('Stock Price Prediction Using Linear Regression')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()

    # Show plot
    plt.show()

# regression_plot("INTC")
# lstm_plot("INTC")

# regression_plot_predictions("INTC")
# lstm_plot_predictions("INTC")