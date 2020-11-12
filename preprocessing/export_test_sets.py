#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

def export_df(mth=1):
    if mth != 1 and mth != 6:
        raise Exception("Months can either be only 1 or 6.")
    filenames = [i.replace('.csv', '') for i in sorted(os.listdir(f'{mth}mo_filtered'))]
    for f in filenames:
        print(f'{mth}-month: {f}')
        df = pd.read_csv(f'{mth}mo_filtered/{f}.csv').sort_values('date').reset_index(drop=True)
        eighty_len = int(len(df)*0.8)
        eighty_df = df[:eighty_len]
        twenty_df = df[eighty_len:]
        eighty_df.to_csv(f'{mth}mo_train_val_data/{f}.csv', index = False)
        twenty_df.to_csv(f'{mth}mo_test_data/{f}.csv', index = False)

export_df(1)

