import pandas as pd
import numpy as np
import finnhub
import os
from dotenv import load_dotenv
from urllib.request import urlopen
from urllib.request import Request
from datetime import datetime, timedelta

load_dotenv()
api_key = os.getenv('SECRET_KEY')

from datetime import datetime, timedelta

from datetime import datetime, timedelta

def get_dates():

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)


    date_ranges = []
    current_start = start_date
    use_6_day_interval = True  


    for _ in range(60):

        range_length = timedelta(days=6 if use_6_day_interval else 7)
        current_end = current_start + range_length - timedelta(days=1)
        

        if current_end > end_date:
            current_end = end_date
        
        date_ranges.append((current_start.strftime('%Y-%m-%d'), current_end.strftime('%Y-%m-%d')))
        
        current_start = current_end + timedelta(days=1)
        
        if current_start > end_date:
            break
        
        use_6_day_interval = not use_6_day_interval

    return date_ranges





def news(stock):
    global api_key
    finnhub_client = finnhub.Client(api_key=api_key)
    from_date = ''
    to_date = ''
    news_arr = []
    call_count = 60
    dates = get_dates()
    for i in dates:
        from_date = i[0]
        to_date = i[1]
        res = finnhub_client.company_news(stock, _from=from_date, to=to_date)
        df = pd.DataFrame(res) \
                            .rename(columns={
                                'category': 'category',
                                'datetime': 'timestamp',
                                'headline': 'headline',
                                'id': 'id',
                                'image': 'image',
                                'related': 'related',
                                'source' : 'source',
                                'summary' : 'summary',
                                'url' : 'url'
                                }) \
                            .set_index(keys = 'timestamp')
        df.index = pd.to_datetime(df.index, unit='s')
        news_arr.append(df)
    final_df = pd.concat(news_arr)
    return final_df

headlines = news('AAPL')
print(headlines.shape)

print(headlines.head())

print(headlines.tail())

