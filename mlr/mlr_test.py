#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

DIVIDER = '-'*50

# Cleaned up data sorted by date, with zero / nan columns removed such that all companies have same number of features.
ONE_MONTH_TRAIN_VAL_DIR = '../Data With Quarters/1mo_train_val_data_with_quarters'
SIX_MONTH_TRAIN_VAL_DIR = '../Data With Quarters/6mo_train_val_data_with_quarters'
ONE_MONTH_TEST_DIR = '../Data With Quarters/1mo_test_data_with_quarters'
SIX_MONTH_TEST_DIR = '../Data With Quarters/6mo_test_data_with_quarters'

def get_df_with_removed_duplicate_blocks(df):
  i, prev, indices = 0, -1, []
  while i < len(df)-1 and prev != i:
    prev = i
    curr_qtr, next_qtr = df.iloc[i]['quarter'], df.iloc[i+1]['quarter']
    curr_row, next_row = df.iloc[i, 3:-4], df.iloc[i+1, 3:-4]
    if curr_qtr != next_qtr and (curr_row == next_row).all():
      monitored_qtr = next_qtr
      while monitored_qtr == next_qtr and (curr_row == next_row).all():
        indices.append(next_row.name)
        i += 1
        if i < len(df)-1:
            next_qtr, next_row = df.iloc[i+1]['quarter'], df.iloc[i+1, 3:-4]
        else:
            break
      i -= 1
    else:
      i += 1
    
  return df[~df.index.isin(indices)]

def remove_outliers(df):
  for i in range(len(df)):
    if (abs(df.loc[[i]]._get_numeric_data()) > 1e15).any().any():
      df.drop(i, axis=0, inplace=True)

def print_and_log(string=''):
  print(string)
  LOG_FILE.write(string + '\n')

def get_days_diff_series(df):
  date_series = df['date'].apply(pd.to_datetime)
  diff_ls = [1]
  for i in range(len(date_series) - 1):
    diff = (date_series[i+1] - date_series[i]).days
    diff_ls.append(diff + diff_ls[i])
  return pd.Series(diff_ls)

def init_df(directory, filename, addTimeColumns=True):
  df = pd.read_csv(f'{directory}/{filename}')
  if addTimeColumns:
      df['t1'] = get_days_diff_series(df)
      df['t2'] = df['t1'] ** 2
      df['t3'] = df['t1'] ** 3
  remove_outliers(df)  
  df = get_df_with_removed_duplicate_blocks(df)
  dataframes[filename] = df
  return df

def run_final_iteration(xs, num_months=1):
  if num_months not in [1, 6]:
    raise Exception('num_months can only be 1 or 6.')
  TRAIN_VAL_DIR = ONE_MONTH_TRAIN_VAL_DIR if num_months == 1 else SIX_MONTH_TRAIN_VAL_DIR
  TEST_VAL_DIR = ONE_MONTH_TEST_DIR if num_months == 1 else SIX_MONTH_TEST_DIR
  category = f'{num_months} months'
  sorted_files = sorted(os.listdir(TRAIN_VAL_DIR))
  print_and_log(f'{DIVIDER} Training {category} for {len(sorted_files)} companies, using {len(xs)} features: {xs} {DIVIDER}')
  rmses = []
  companies_models_map = {}
  # feature_to_vals = {feature: [] for feature in xs}
  min_rmse_company, min_rmse = None, None
  for k, f in enumerate(sorted_files):
    # Initialise dataframe
    training_df = init_df(TRAIN_VAL_DIR, f)
    validation_df = init_df(TEST_VAL_DIR, f)

    # Generate equation
    y = training_df.columns[1]
    eqn = y + ' ~ ' + ' + '.join(xs)

    # training of model
    model = ols(eqn, training_df)
    result = model.fit()
    companies_models_map[f] = model

    pr = sm.stats.anova_lm(result, typ=2).iloc[:-1,:]['PR(>F)'] 
    # for index in pr.index:
    #   feature_to_vals[index].append(pr[index])

    # Validation RMSE
    y_hat = result.predict(validation_df)
    rmse = ((validation_df[y] - y_hat)**2).mean() ** 0.5
    if min_rmse is None or rmse < min_rmse:
      min_rmse_company, min_rmse = f, rmse

    rmses.append(rmse)
    print_and_log(f'{k+1}. {f}; train_num_rows: {len(training_df)}; val_num_rows: {len(validation_df)}; rmse: {rmse}')

  # feature_to_average_f_val_map = {k: sum(v)/len(v) for k, v in feature_to_vals.items()}
  average_rmse = sum(rmses) / len(rmses)  

  # return feature_to_average_f_val_map, average_rmse
  return companies_models_map, average_rmse, min_rmse_company, min_rmse

LOG_FILE = open('final_train_test_1mo.txt', 'w+')
one_mo_xs = ['roic', 'returnOnCapitalEmployed', 'priceToOperatingCashFlowsRatio', 'priceCashFlowRatio'] # Change optimal features here
one_mo_companies_models_map, one_mo_average_rmse, one_mo_min_rmse_company, one_mo_min_rmse = run_final_iteration(one_mo_xs, num_months=1)
LOG_FILE.close()

LOG_FILE = open('final_train_test_6mo.txt', 'w+')
six_mo_xs = ['operatingIncomeGrowth'] # Change optimal features here
six_mo_companies_models_map, six_mo_average_rmse, six_mo_min_rmse_company, six_mo_min_rmse = run_final_iteration(six_mo_xs, num_months=6)
LOG_FILE.close()

LOG_FILE = open('final_train_test_1mo_2.txt', 'w+')
one_mo_xs_2 = ['quarter'] # Change optimal features here
one_mo_companies_models_map_2, one_mo_average_rmse_2, one_mo_min_rmse_company_2, one_mo_min_rmse_2 = run_final_iteration(one_mo_xs_2, num_months=1)
LOG_FILE.close()

LOG_FILE = open('final_train_test_6mo_2.txt', 'w+')
six_mo_xs_2 = ['netIncomeGrowth', 'operatingCashFlowGrowth', 'evToOperatingCashFlow', 'netDebtToEBITDA', 'quickRatio', 'cashConversionCycle', 'longTermDebtToCapitalization', 'freeCashFlowOperatingCashFlowRatio'] # Change optimal features here
six_mo_companies_models_map_2, six_mo_average_rmse_2, six_mo_min_rmse_company_2, six_mo_min_rmse_2 = run_final_iteration(six_mo_xs_2, num_months=6)
LOG_FILE.close()

def plt_graph(f, model_map, plt_directory, num_months=1):
    if num_months not in [1, 6]:
        raise Exception('num_months can only be 1 or 6.')
    TRAIN_VAL_DIR = ONE_MONTH_TRAIN_VAL_DIR if num_months == 1 else SIX_MONTH_TRAIN_VAL_DIR
    TEST_VAL_DIR = ONE_MONTH_TEST_DIR if num_months == 1 else SIX_MONTH_TEST_DIR
    y = 'closePlusOneMonth' if num_months == 1 else 'closePlusSixMonth'
    validation_df = init_df(TEST_VAL_DIR, f)
    actual = validation_df[y]
    x = validation_df['t1']
    fits = model_map[f].fit().predict(validation_df)
    plt.plot(x, actual, label='Real Stock Price', c='#414141')
    plt.plot(x, fits, label = 'Predicted Stock Price', c='#78ae6e')
    plt.xlabel('Time', fontsize='12')
    plt.xlabel('Time', fontsize='12')
    plt.ylabel('Price', fontsize='12')
    plt.title('Stock Price Prediction', fontsize='13')
    plt.legend(loc='upper left')
    filename = f.replace('.csv', '')
    plt.savefig(f'{plt_directory}/{filename}.png')
    plt.clf()

def plt_graphs_1mo(f='CVGW.csv'):
    plt_graph(f, one_mo_companies_models_map, '1mo_4_features', num_months=1)
    plt_graph(f, one_mo_companies_models_map_2, '1mo_1_feature', num_months=1)

def plt_graphs_6mo(f='CVGW.csv'):
    plt_graph(f, six_mo_companies_models_map,'6mo_1_feature', num_months=6, )
    plt_graph(f, six_mo_companies_models_map_2, '6mo_8_features', num_months=6)

for k, f in enumerate(sorted(os.listdir(ONE_MONTH_TRAIN_VAL_DIR))):
    print(k, f)
    plt_graphs_1mo(f)

for k, f in enumerate(sorted(os.listdir(SIX_MONTH_TRAIN_VAL_DIR))):
    print(k, f)
    plt_graphs_6mo(f)


# <!-- run_iteration(xs, num_months=1) -->
