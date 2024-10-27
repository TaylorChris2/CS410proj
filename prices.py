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

aapl = yf.Ticker("AAPL")
hist = aapl.history("1y")

plt.plot(hist["Open"])
plt.show()