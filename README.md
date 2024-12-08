# CS410proj

## Environment Setup
   Navigate to the source directory and create a python virtual environment using
   
   ~~~python
   python3 -m venv ./venv
   ~~~

  To activate the environment, in the same directory run 

  ~~~python
  .venv/Scripts/activate/ (for Windows)
  source .venv/bin/activate (for Mac)

  ~~~
More information on creating a virtual environment and installing requirements is present [here](https://frankcorso.dev/setting-up-python-environment-venv-requirements.html) 
  
  Then create a new file in your source directory called ` .env `

  Go to [Finnhub](https://finnhub.io/register) and get an api key which you should put in your ` .env ` folder in the following manner

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
This will generate 4 graphs for which the explanations are below
