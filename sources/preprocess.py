#!/usr/bin/python

import numpy as np
import pandas as pd


def default(row):
    if row['loss'] > 0:
        return 1
    return 0

data = pd.read_csv('train_v2.csv').head(2000)

data['default'] = data.apply(default, axis=1)
keep_col = []
for i in range(1, 53):
    if i != 11 and i != 12: # for whatever reason these features are missing
        keep_col.append('f' + str(i))
keep_col.append('default')
data = data[keep_col]

data.iloc[np.r_[0:1000]].to_csv('train.csv', sep=',', header=False, index=False)
data.iloc[np.r_[0:1000]].to_csv('train_schema.csv', sep=',')
data.iloc[np.r_[1000:1500]].to_csv('validation.csv', sep=',', header=False, index=False)
data.iloc[np.r_[1000:1500]].to_csv('validation_schema.csv', sep=',')
data.iloc[np.r_[1500:2000]].to_csv('test.csv', sep=',', header=False, index=False)
data.iloc[np.r_[1500:2000]].to_csv('test_schema.csv', sep=',')

data.iloc[np.r_[0:500]].to_csv('train500.csv', sep=',', header=False, index=False)
data.iloc[np.r_[0:250]].to_csv('train250.csv', sep=',', header=False, index=False)
data.iloc[np.r_[0:100]].to_csv('train100.csv', sep=',', header=False, index=False)
data.iloc[np.r_[0:50]].to_csv('train50.csv', sep=',', header=False, index=False)