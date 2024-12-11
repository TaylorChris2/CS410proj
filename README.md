# CS410proj

##   We strongly recommend using Vscode. Make sure you change your current working directory to the project directory using cd in terminal of Vscode

## Environment Setup (Strongly recommend this. You can use your global environment if you are more comfortable with that)
   Navigate to the source directory and create a python virtual environment using
   
   ~~~python
   python3 -m venv ./venv
   ~~~

  To activate the environment, in the same directory run

  ~~~python
  venv/Scripts/activate/ (for Windows)
  source venv/bin/activate (for Mac)

  ~~~
Your terminal prompts will then change to show the virtual environment's name (venv)

More information on creating a virtual environment and installing requirements is present [here](https://frankcorso.dev/setting-up-python-environment-venv-requirements.html) 
  
  Then create a new file in your source directory called ` .env `

  Go to [Finnhub](https://finnhub.io/register) and get an api key which you should put in your ` .env ` folder in the following manner. You can use the free tier. 

  ~~~
    SECRET_KEY = "the key you got from Finnhub"
  ~~~

## Run the Code
   Navigate to the root directory of the project

   ~~~python
   pip install -r "requirements.txt"
   ~~~

   Then run 

   ~~~python
   python3 ./main.py
   or
   python ./main.py

   ~~~
This will generate 7 plots. And will print out RMSE and MAE values for LSTM and Linear Regression in the terminal

The plots that will be outputted are described below, all values are in relation to the user defined stock on line 40:
1. Normalized stock prices and sentiment values over the past year
2. Percentage of positive, neutral, and negative documents over the past year
3. Actual stock values over the past year and stock values predicted by our linear regression model over the past year
4. Actual stock values over the past year and stock values predicted by our LSTM model over the past year
5. Actual stock values over the past year and stock values predicted by our linear regression model for only our test data
6. Actual stock values over the past year and stock values predicted by our LSTM model for only our test data
7. Actual stock values, stock values predicted by our linear regression model, and stock values predicted by our LSTM model over only our test data

## References 
   - https://frankcorso.dev/setting-up-python-environment-venv-requirements.html
   - https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233
   - https://ibkrcampus.com/ibkr-quant-news/exploring-the-finnhub-io-api/
   - Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social       Media (ICWSM-14). Ann Arbor, MI, June 2014.
