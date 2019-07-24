import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
pd.set_option('max_columns', 100)
# print(train_data.shape)
# print(test_data.shape)
full_data = pd.concat([train_data, test_data], ignore_index=True)
print(full_data.shape)
temp = full_data.isnull().sum()
print(temp[temp>0].sort_values(ascending=False))
# print(full_data.describe())