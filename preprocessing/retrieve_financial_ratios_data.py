#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import numpy as np
import pandas as pd
from urllib.request import urlopen

datasets=["NYSE.txt","NASDAQ.txt"]

tickers=[]
for dataset in datasets:
    print(dataset)
    file = open(dataset, "r")
    while True:
        line=file.readline()
        if not line:
            break
        temp=(line.split(','))
        tickers.append(temp[0])
        
tickers.sort()
print(tickers)

categories_quarterly = ['ratios', 'income-statement', 
              'balance-sheet-statement', 
              'cash-flow-statement', 
              'income-statement-as-reported', 
              'balance-sheet-statement-as-reported', 
              'cash-flow-statement-as-reported', 
              'financial-statement-full-as-reported', 
              'enterprise-values', 
              'key-metrics', 
              'financial-growth']
              
categories_without_suffix = ['historical-rating', 
                             'historical-daily-discounted-cash-flow', 
                             'historical-market-capitalization']

def get_jsonparsed_data(url):
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

def get_df(active_ticker, category, include_period=True):
    url = f'https://financialmodelingprep.com/api/v3/{category}/{active_ticker}'
    url += '?period=quarter&' if include_period else '?'
    url += 'apikey=sample_api_key'
    data = get_jsonparsed_data(url)
    df = pd.DataFrame(data, index=[i for i in range(len(data))])
    return df

def write_data(category, include_period=True, start=0):
    if category == 'key-metrics':
        sliced_tickers = tickers[tickers.index('MFO'):]
    elif category == 'income-statement':
        sliced_tickers = tickers[tickers.index('CSX'):]
    elif category == 'historical-rating':
        sliced_tickers = tickers[tickers.index('AOSL'):]
    else:
        sliced_tickers = tickers
        
    for active_ticker in sliced_tickers:
        print(f'Fetching data for {active_ticker}...')
        df = call_api(lambda: get_df(active_ticker, category, include_period))
        if df.empty and ("-" in active_ticker or "." in active_ticker):
            active_ticker = (active_ticker.replace("-", ".")
                             if "-" in active_ticker else active_ticker.replace(".", "-"))
            df = call_api(lambda: get_df(active_ticker, category, include_period))
        if df.empty and ("-" in active_ticker or "." in active_ticker):
            delimiter = "." if "." in active_ticker else "-"
            active_ticker = active_ticker[:active_ticker.find(delimiter)]
            df = call_api(lambda: get_df(active_ticker, category, include_period))
        if df.empty:
            print(f'Error: Data not found for {active_ticker}.')
        else:
            df.to_csv(f"{category}/{active_ticker}.csv",header=True,index=True)
            print(f'Data successfully written for {active_ticker}.')

line='-'*100

for i in categories_quarterly:
    if i == 'ratios':
        continue
    print(line)
    print(i)
    print(line)
    write_data(i)

for j in categories_without_suffix:
    print(line)
    print(j)
    print(line)
    write_data(j, include_period=False)

