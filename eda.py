import pandas as pd
import numpy as np
import finnhub
import os
from dotenv import load_dotenv
from urllib.request import urlopen
from urllib.request import Request


load_dotenv()
api_key = os.getenv('SECRET_KEY')


