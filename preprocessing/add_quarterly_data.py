#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

def export_with_quarters(directory):
    filenames = [i.replace('.csv', '') for i in sorted(os.listdir(directory))]
    for f in filenames:
        print(f)
        df = pd.read_csv(f'{directory}/{f}.csv')
        df['quarter'] = df['quarter'].astype('category')
        df['quarter'] = (pd.to_datetime(df['date']).dt.month-1)//3+1
        new_directory = f'{directory}_with_quarters'
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
        df.to_csv(f'{new_directory}/{f}.csv', index = False)

directories = ['1mo_train_val_data_w_quarters', '1mo_test_data_w_quarters', '6mo_train_val_data_w_quarters', '6mo_test_data_w_quarters']

for directory in directories:
    print('-'*100 + directory + '-'*100)
    export_with_quarters(directory)

