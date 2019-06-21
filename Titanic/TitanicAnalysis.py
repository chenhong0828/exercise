import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

data_train = pd.read_csv('E:/Titanic/train.csv')
# print(data_train[:5])
# print(data_train.info())
# print(data_train.describe())

fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u'获救情况 (1为获救)')
plt.ylabel(u'人数')

# plt.subplot2grid((2,3),(0,1))
# data_train.Pclass.vlaue_counts().plot(kind='bar')
# plt.ylabel(u'人数')
# plt.title(u'乘客等级分布')

plt.subplot2grid((2,3),(0,1))#需修改，考虑循环
x = data_train.Pclass
y1 = (data_train.Survived == 1)[data_train.Pclass == 1].sum(axis=0)
y2 = (data_train.Survived == 0)[data_train.Pclass == 1].sum(axis=0)
plt.bar(x, y1, color='green', label='获救')
plt.bar(x, y2, bottom=y1, color='red', label='未获救')
plt.legend(loc=[1, 0])

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u'年龄')
plt.grid(b=True, which='major',axis='y')
plt.title(u'按年龄看获救分布（1为获救）')

plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u'年龄')
plt.ylabel(u'密度')
plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')


plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u'各登船口岸上传人数')
plt.ylabel(u'人数')
plt.show()
