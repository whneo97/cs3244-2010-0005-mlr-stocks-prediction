#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

# Maps companies to names of zero columns
dictionary = {}
initial_total_cols = None

def add_to_dict(key, val):
    try:
        dictionary[key].add(val)
    except:
        dictionary[key] = {val}

def getDfWithNonZeroLabels(df, month=6):
    labelHeader = 'closePlus' + ('Six' if month == 6 else 'One') + 'Month'
    otherLabelHeader = 'closePlus' + ('One' if month == 6 else 'Six') + 'Month'
    cols = list(df.columns.values)
    cols.pop(cols.index(labelHeader))
    cols.pop(cols.index('date'))
    df = df[['date', labelHeader] + cols]
    return df[df[labelHeader] != 0].drop(otherLabelHeader, axis=1)

def add_col_name_of_entirely_zero_columns(df, company_name):
    global initial_total_rows, initial_total_cols
    print(f'Identifying columns from {company_name}...')

    columns = df.columns
    if initial_total_cols is None:
        initial_total_cols = len(columns)
    else:
        assert initial_total_cols == len(columns)
        
    for col in columns:
        if df[col].dtype != np.object and df[col].dtype.name != 'category' and (df[col] == 0).any():
            add_to_dict(company_name, col)
        if df[col].isnull().any():
            add_to_dict(company_name, col)

dataframes = []

filenames = [i.replace('.csv', '') for i in sorted(os.listdir('../combined-all-processed-y'))]
for f in filenames:
    df = getDfWithNonZeroLabels(pd.read_csv(f'../combined-all-processed-y/{f}.csv'), 1)
    if df.empty or int(len(df)*0.8) <= 60 or len(df) - int(len(df)*0.8) <= 60:
        continue
    add_col_name_of_entirely_zero_columns(df, f)
    dataframes.append(f)

# def get_cost(n_cols_removed, n_rows_removed, n_union_zero_cols, 
#               initial_total_rows, initial_total_cols):
#     col_cost = (n_cols_removed + n_union_zero_cols) / initial_total_cols
#     row_cost = (n_rows_removed) / initial_total_rows
#     return abs(row_cost - col_cost)

def get_cost(n_cols_removed, n_rows_removed, n_union_zero_cols, 
              initial_total_rows, initial_total_cols):
    col_cost = (n_cols_removed + n_union_zero_cols) / initial_total_cols
    row_cost = (n_rows_removed) / initial_total_rows
    return (row_cost + col_cost)/2

def get_union_zero_cols(new_dict):
    new_set = set()
    for v in new_dict.values():
        for u in v:
            new_set.add(u)
    return new_set

def remove_rows_and_cols(new_dict, next_row_to_remove, next_col_to_remove):
    if next_row_to_remove != "":
        del new_dict[next_row_to_remove]
    if next_col_to_remove != "":
        for v in new_dict.values():
            v.discard(next_col_to_remove)

glob_rows_removed = []
glob_cols_removed = []
glob_cost = []
glob_points = ()

feature_counts = {}
for feature in get_union_zero_cols(dictionary):
    feature_count = sum(list(v).count(feature) for v in dictionary.values())
    feature_counts[feature] = feature_count

def plot_graph():
    sorted_indices = np.argsort(glob_rows_removed)
    x = np.array(glob_rows_removed)[sorted_indices]
    y1 = np.array(glob_cols_removed)[sorted_indices]
    y2 = np.array(glob_cost)[sorted_indices]
    opt_x, opt_y = glob_points
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel('#Companies Removed', fontsize=50, fontweight='bold', color='black')
    ax1.set_ylabel('#Columns Removed', fontsize=50, fontweight='bold', color=color)
    ax1.plot(x, y1, c=color, linewidth=20)
    ax1.plot(opt_x, opt_y, marker='x', c='green', mew=5, ms=50, linewidth=20)
    ax1.plot([opt_x, opt_x], [0, opt_y], linestyle='dashed', c='black', linewidth=10)
    ax1.plot([0, opt_x], [opt_y, opt_y], linestyle='dashed', c='black')
    ax1.text(opt_x + 100, opt_y + 4, f'({opt_x}, {opt_y})', fontsize=50)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(color=color, alpha=0.5)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Total Cost', color=color, fontsize=50, fontweight='bold')
    ax2.plot(x, y2, c=color ,linewidth=20)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(color=color, alpha=0.5)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped    
    plt.rcParams["figure.figsize"] = (20,15)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

def plot_graph():
    sorted_indices = np.argsort(glob_rows_removed)
    x = np.array(glob_rows_removed)[sorted_indices]
    y1 = np.array(glob_cols_removed)[sorted_indices]
    y2 = np.array(glob_cost)[sorted_indices]
    opt_x, opt_y = glob_points
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel('#Companies Removed', fontsize=25, fontweight='bold', color='black')
    ax1.set_ylabel('#Columns Removed', fontsize=25, fontweight='bold', color=color)
    ax1.plot(x, y1, c=color)
    ax1.plot(opt_x, opt_y, marker='x', c='green', mew=5, ms=30)
    ax1.plot([opt_x, opt_x], [0, opt_y], linestyle='dashed', c='black')
    ax1.plot([0, opt_x], [opt_y, opt_y], linestyle='dashed', c='black')
    ax1.text(opt_x + 100, opt_y + 4, f'({opt_x}, {opt_y})', fontsize=25)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(color=color, alpha=0.5)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Total Cost', color=color, fontsize=25, fontweight='bold')
    ax2.plot(x, y2, c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(color=color, alpha=0.5)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped    
    plt.rcParams["figure.figsize"] = (20,15)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

def remove_greedily(dictionary):    
    global glob_rows_removed, glob_cols_removed, glob_cost, glob_points
    glob_rows_removed = []
    glob_cols_removed = []
    glob_cost = []
    glob_points
    initial_total_rows = len(filenames)
    
    def add_to_global_variables(cost, cols_removed, rows_removed):
        global glob_rows_removed, glob_cols_removed, glob_cost
        print('Cost:', cost, 'Columns removed:', len(cols_removed), 'Companies removed: ', len(rows_removed))
        glob_rows_removed.append(len(rows_removed))
        glob_cols_removed.append(len(cols_removed))
        glob_cost.append(cost)
    
#     rows_to_remove = sorted(list(dictionary.keys()), 
#                             key=lambda x: len(dictionary[x]), reverse=True)
    rows_to_remove = sorted(list(dictionary.keys()), 
                            key=lambda x: min(feature_counts[f] for f in dictionary[x]))
    
    new_dict = deepcopy(dictionary)
    rows_removed = set()
    cols_removed = get_union_zero_cols(new_dict)
    min_cost = get_cost(0, len(rows_removed), len(cols_removed), 
                        initial_total_rows, initial_total_cols)
    min_cost_attr = (new_dict, rows_removed, cols_removed, min_cost)
    add_to_global_variables(min_cost, cols_removed, rows_removed)
    
    for row_to_remove in rows_to_remove:
        remove_rows_and_cols(new_dict, row_to_remove, '')
        rows_removed.add(row_to_remove)
        cols_removed = get_union_zero_cols(new_dict)
        new_cost = get_cost(0, len(rows_removed), len(cols_removed), 
                            initial_total_rows, initial_total_cols)
        add_to_global_variables(new_cost, cols_removed, rows_removed)
        
        if new_cost < min_cost_attr[-1]:
            min_cost_attr = (new_dict.copy(), rows_removed.copy(), cols_removed.copy(), new_cost)
            glob_points = len(rows_removed), len(cols_removed)           
    return min_cost_attr

new_dict, rows_removed, cols_removed, min_cost = remove_greedily(dictionary)

plot_graph()

companies_to_remove_file = open('companies_to_remove.txt', 'w+')
companies_to_remove_file.write('\n'.join(sorted(rows_removed)))
companies_to_remove_file.close()

columnns_to_remove_file = open('columns_to_remove.txt', 'w+')
columnns_to_remove_file.write('\n'.join(sorted(cols_removed)))
columnns_to_remove_file.close()

