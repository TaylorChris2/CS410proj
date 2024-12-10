# CS410proj

##   Make sure you change your current working directory to the project directory using cd in terminal

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
    API_KEY = "the key you got from Finnhub"
  ~~~

## Run the Code
   Navigate to the folder where the project is located and then run the following command in your terminal

   ~~~python
   pip install -r "requirements.txt"
   ~~~

   Naviage to the main.py file and then run 

   ~~~python
   python3 ./main.py
   ~~~
This will generate 5 graphs. And will print out RMSE and MAE values for LSTM and Linear Regression in the terminal


## References 
   - https://frankcorso.dev/setting-up-python-environment-venv-requirements.html
   - https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233
   - https://ibkrcampus.com/ibkr-quant-news/exploring-the-finnhub-io-api/
   - Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social       Media (ICWSM-14). Ann Arbor, MI, June 2014.
