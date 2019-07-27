import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import seaborn as sns

train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
pd.set_option('max_columns', 100)
# print(train_data.shape)
# print(test_data.shape)

# 探索各数值型变量与因变量的关系
# fig =plt.scatter(train_data['GrLivArea'], train_data.SalePrice)
# plt.xlabel('GrLivArea')
# plt.ylabel('SalePrice')
# plt.show()
# plt.savefig('./GrLivArea_SalePrice.png')
# print(train_data[train_data.BsmtFinSF1 == 5644])

# 对关系明显的全部绘制出散点图
# plt.figure(figsize=(20,10))
# plt.subplots_adjust(wspace=0.2, hspace=0.2)
# sns.set(font_scale=1.1)
# plt.subplot(231)
# sns.scatterplot(train_data['LotArea'], train_data.SalePrice)
# plt.subplot(232)
# sns.scatterplot(train_data['BsmtFinSF1'], train_data.SalePrice)
# plt.subplot(233)
# sns.scatterplot(train_data['TotalBsmtSF'], train_data.SalePrice)
# plt.subplot(234)
# sns.scatterplot(train_data['1stFlrSF'], train_data.SalePrice)
# plt.subplot(235)
# sns.scatterplot(train_data['GrLivArea'], train_data.SalePrice)
# plt.savefig('./OutlierAnalysis.png', dpi=600, bbox_inches='tight')

# 剔除异常值
train_data.drop(train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 200000)].index, inplace=True)

# 剔除异常值后再绘制散点图
# plt.figure(figsize=(20,10))
# plt.subplots_adjust(wspace=0.2, hspace=0.2)
# sns.set(font_scale=1.1)
# plt.subplot(231)
# sns.scatterplot(train_data['LotArea'], train_data.SalePrice)
# plt.subplot(232)
# sns.scatterplot(train_data['BsmtFinSF1'], train_data.SalePrice)
# plt.subplot(233)
# sns.scatterplot(train_data['TotalBsmtSF'], train_data.SalePrice)
# plt.subplot(234)
# sns.scatterplot(train_data['1stFlrSF'], train_data.SalePrice)
# plt.subplot(235)
# sns.scatterplot(train_data['GrLivArea'], train_data.SalePrice)
# plt.savefig('./OutlierAnalysis_new.png', dpi=600, bbox_inches='tight')

# 填充训练集中的缺失值
# temp = train_data.isnull().sum()
# print(temp[temp>0].sort_values(ascending=False))
cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageYrBlt', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrType']
for col in cols:
    train_data[col].fillna('None', inplace=True)
train_data['MasVnrArea'].fillna(0, inplace=True)
train_data['Electrical'].fillna(train_data.Electrical.mode()[0], inplace=True)
# 填充LotFrontage，划分LotArea，用中值填充

def set_missing_LotFrontage(df):
    df['LotAreaCut'] = pd.qcut(df['LotArea'], 10)
    df['LotFrontage'] = df.groupby(['LotAreaCut'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    return df

set_missing_LotFrontage(train_data)
# print(train_data.isnull().sum().sort_values(ascending=False))

NumStr = ['MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
        'GarageYrBlt', 'MoSold', 'YrSold']
train_data.MSSubClass.astype(str)
tem = train_data.groupby(['MSSubClass'])[['SalePrice']].agg(['mean', 'median', 'count']).sort_values(by=[('SalePrice', 'mean'), ('SalePrice', 'median')])

print(tem)
