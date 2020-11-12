import os
import pandas as pd
import numpy as np
import sklearn
import statsmodels.api as sm
import re
from math import ceil
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from time import time

DIR = 'combined-all-processed-y'

def fillQuarters(df):
    df["quarter"] = np.nan
    i = 0
    q = 1
    df.loc[i, 'quarter'] = q        
    while(i < len(df)-1):
        if (df.iloc[i][3:-3] != df.iloc[i+1][3:-3]).any():
            q = (q + 1) if q < 4 else 1
        df.loc[i+1, 'quarter'] = str(q)       
        i += 1
    df['quarter'] = df['quarter'].astype('category')

def remove_entirely_zero_columns(df):
    columns = df.columns
    for col in columns:
        if df[col].dtype != np.object and df[col].dtype.name != 'category' and (df[col] == 0).all():
            df.drop(col, axis=1, inplace=True)
    return df

def initDf(ticker='A', addTimeColumns=False, addQuarters=False):
    df = pd.read_csv(DIR + f'/{ticker}.csv').sort_values('date').reset_index(drop=True)
    if addQuarters:
        fillQuarters(df)
    if addTimeColumns:
        df['t1'] = df.index + 1
        df['t2'] = df['t1'] ** 2
        df['t3'] = df['t1'] ** 3
    df['ticker'] = ticker
    cols = list(df.columns.values)
    cols.pop(cols.index('ticker'))
    df = df[['ticker'] + cols]
    return df

def getDfWithNonZeroLabels(df, month=6):
    labelHeader = 'closePlus' + ('Six' if month == 6 else 'One') + 'Month'
    otherLabelHeader = 'closePlus' + ('One' if month == 6 else 'Six') + 'Month'
    cols = list(df.columns.values)
    cols.pop(cols.index(labelHeader))
    cols.pop(cols.index('date'))
    df = df[['date', labelHeader] + cols]
    return remove_entirely_zero_columns(df[df[labelHeader] != 0].drop(otherLabelHeader, axis=1))

def getTrainingVerificationDf(df, split=0.75):
    return df[:int(split * len(df))]

def get_company_df(ticker, month=1):
    df = initDf(ticker)
    return getDfWithNonZeroLabels(df, month)

filenames = [i.replace('.csv', '') for i in sorted(os.listdir('combined-all-processed-y'))]

frames = []
for f in filenames:
    print(f)
    dataframe = get_company_df(f, 6)
    if not dataframe.empty:
        frames.append(dataframe)

result = pd.concat(frames)

sorted_result = result.sort_values('date').reset_index(drop=True)
n = len(sorted_result)
train_index = int(n * 0.8)
train_df = sorted_result[:train_index]
test_df = sorted_result[train_index:]

for f in filenames:
    print(f)
    out_df = train_df[train_df['ticker'] == f]
    out_df.drop('ticker', axis=1, inplace=True)
    out_df.to_csv(f'6mo/{f}.csv', index=False)
    
test_df.to_csv('6mo_test.csv', index=False)
