import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sns

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
pd.set_option('max_columns', 100)
# print(train_data.shape)
# print(test_data.shape)

# fig =plt.scatter(train_data['GrLivArea'], train_data.SalePrice)
# plt.xlabel('GrLivArea')
# plt.ylabel('SalePrice')
# plt.show()
# plt.savefig('./GrLivArea_SalePrice.png')
# print(train_data[train_data.BsmtFinSF1 == 5644])

plt.figure(figsize=(20,10))
plt.subplots_adjust(wspace=0.2, hspace=0, top=0.3, bottom=0.2)
sns.set(font_scale=1.1)
plt.subplot(231)
sns.scatterplot(train_data['LotArea'], train_data.SalePrice)
plt.subplot(232)
sns.scatterplot(train_data['BsmtFinSF1'], train_data.SalePrice)
plt.subplot(233)
sns.scatterplot(train_data['TotalBsmtSF'], train_data.SalePrice)
plt.subplot(234)
sns.scatterplot(train_data['1stFlrSF'], train_data.SalePrice)
plt.subplot(235)
sns.scatterplot(train_data['GrLivArea'], train_data.SalePrice)
plt.savefig('./OutlierAnalysis.png', dpi=600)

# plt.show()

# full_data = pd.concat([train_data, test_data], ignore_index=True)
# print(full_data.shape)



# temp = full_data.isnull().sum()
# print(temp[temp>0].sort_values(ascending=False))

# print(full_data.describe())