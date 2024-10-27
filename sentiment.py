import pandas as pd
import numpy as np
import finnhub
import os
from dotenv import load_dotenv
from urllib.request import urlopen
from urllib.request import Request
from datetime import datetime, timedelta


stock = 'AAPL'
directory_name = 'data'
filename = f"{stock}.csv"
file_path = os.path.join(directory_name, filename)
df = pd.read_csv(file_path)
print(df.head())
print("Test")