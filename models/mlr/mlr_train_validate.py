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
LOG_FILE = open('mlr_train_validate.txt', 'w+')

# Cleaned up data sorted by date, with zero / nan columns removed such that all companies have same number of features.
ONE_MONTH_TRAIN_VAL_DIR = '../Data With Quarters/1mo_train_val_data_with_quarters'
SIX_MONTH_TRAIN_VAL_DIR = '../Data With Quarters/6mo_train_val_data_with_quarters'
ONE_MONTH_TEST_DIR = '../Data With Quarters/1mo_test_data_with_quarters'
SIX_MONTH_TEST_DIR = '../Data With Quarters/6mo_test_data_with_quarters'
DIRECTORIES = [ONE_MONTH_TRAIN_VAL_DIR, SIX_MONTH_TRAIN_VAL_DIR]
TRAIN_DIRECTORIES = [ONE_MONTH_TRAIN_VAL_DIR, SIX_MONTH_TRAIN_VAL_DIR]
TEST_DIRECTORIES = [ONE_MONTH_TEST_DIR, SIX_MONTH_TEST_DIR]

dataframes = {}

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
  try:
    return dataframes[directory + '/' + filename]
  except:
    pass
  df = pd.read_csv(f'{directory}/{filename}')
  if addTimeColumns:
      df['t1'] = get_days_diff_series(df)
      df['t2'] = df['t1'] ** 2
      df['t3'] = df['t1'] ** 3
  remove_outliers(df)  
  df = get_df_with_removed_duplicate_blocks(df)
  dataframes[filename] = df
  return df

def run_iteration(directory, xs):
  category = directory.split('/')[-1].split('_')[0]
  sorted_files = sorted(os.listdir(directory))
  print_and_log(f'{DIVIDER} Training {category} for {len(sorted_files)} companies, using {len(xs)} features: {xs} {DIVIDER}')
  rmses = []
  feature_to_vals = {feature: [] for feature in xs}
  for k, f in enumerate(sorted_files):
    # Initialise dataframe
    df = init_df(directory, f)
    training_df = df[:int(0.8*len(df))]
    validation_df = df[int(0.8*len(df)):]

    # Generate equation
    y = training_df.columns[1]
    eqn = y + ' ~ ' + ' + '.join(xs)

    # training of model
    model = ols(eqn, training_df)
    result = model.fit()

    pr = sm.stats.anova_lm(result, typ=2).iloc[:-1,:]['PR(>F)'] 
    for index in pr.index:
      feature_to_vals[index].append(pr[index])

    # Validation RMSE
    y_hat = result.predict(validation_df)
    rmse = ((validation_df[y] - y_hat)**2).mean() ** 0.5
    rmses.append(rmse)
    print_and_log(f'{k+1}. {f}; train_num_rows: {len(training_df)}; val_num_rows: {len(validation_df)}; rmse: {rmse}')


  feature_to_average_f_val_map = {k: sum(v)/len(v) for k, v in feature_to_vals.items()}
  average_rmse = sum(rmses) / len(rmses)  

  return feature_to_average_f_val_map, average_rmse

def run_for_directory(directory):
    df = init_df(directory, sorted(os.listdir(directory))[0])
    all_features = list(df.columns[2:]) # an original list of all features
    feature_mapping = {} # map tuple(features) to rmse values

    # Comparable features
    # Features are first compared by rmse values, before length of features i.e. number of dimensions
    class FeaturesComparable(object):
        def __init__(self, features):
            self.features = features
            
        def __eq__(self, other):
            return (feature_mapping[self.features] == feature_mapping[other.features] 
                    and len(self.features) == len(other.features))
        def __lt__(self, other):
            return (feature_mapping[self.features] < feature_mapping[other.features] 
                    or (feature_mapping[self.features] == feature_mapping[other.features] 
                        and len(self.features) < len(other.features)))
        def __ne__(self, other):
            return (feature_mapping[self.features] != feature_mapping[other.features] 
                    or len(self.features) != len(other.features)) 
        def __gt__(self, other):
            return (feature_mapping[self.features] > feature_mapping[other.features] 
                    or (feature_mapping[self.features] == feature_mapping[other.features] 
                        and len(self.features) > len(other.features)))
        def __le__(self, other):
            return ((feature_mapping[self.features] < feature_mapping[other.features] 
                    or (feature_mapping[self.features] == feature_mapping[other.features] 
                        and len(self.features) < len(other.features))) or 
                    (feature_mapping[self.features] == feature_mapping[other.features] 
                    and len(self.features) == len(other.features)))
        def __ge__(self, other):
            return ((feature_mapping[self.features] > feature_mapping[other.features] 
                    or (feature_mapping[self.features] == feature_mapping[other.features] 
                        and len(self.features) > len(other.features))) or 
                    (feature_mapping[self.features] == feature_mapping[other.features] 
                    and len(self.features) == len(other.features)))

    while len(all_features) != 0:
        f_val_mapping, average_rmse = run_iteration(directory, all_features) # update mapping of feature name to avg f-stat val at end of iteration
        feature_mapping[tuple(all_features)] = average_rmse
        print_and_log(f'\nAverage rmse: {average_rmse}')

        # remove the smallest f-stat feature from features list
        feature_to_remove = min(f_val_mapping.keys(), key=lambda feature: f_val_mapping[feature])
        min_f_val = f_val_mapping[feature_to_remove]
        print_and_log(f'\nFeature removed: {feature_to_remove}, f_val: {min_f_val}')
        all_features.remove(feature_to_remove)

    # get tuple of features with the smallest rmse
    selected_features = min(feature_mapping.keys(), key=lambda features: FeaturesComparable(features))
    min_rmse = feature_mapping[selected_features]
    print_and_log(f'\n{len(selected_features)} features selected: {selected_features}, {min_rmse}')

    return selected_features, min_rmse

def train_for_all_directories():
  for directory in DIRECTORIES:
    print_and_log(f'\n{DIVIDER} Training for {directory} {DIVIDER}\n')
    run_for_directory(directory)

train_for_all_directories()
LOG_FILE.close()

