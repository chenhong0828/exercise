import numpy as np
import pandas as pd

train_data = pd.read_csv('./train.csv')
pd.set_option('max_columns', 100)
# print(train_data.shape)
# print(train_data.isnull().sum().sort_values(ascending=False))
# print(train_data['PoolQC'].value_counts(dropna=False))
# print(train_data['LotFrontage'].max())
print(train_data.describe())