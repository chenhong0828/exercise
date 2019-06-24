import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats 

sns.set(style="darkgrid", font="SimHei", rc={"axes.unicode_minus": False})

data = pd.read_csv("CompletedDataset.csv")
# print(data.shape)
pd.set_option("max_columns", 20)
# print(data.head())
# print(data.isnull().sum(axis=0))
# print(data.describe())
# sns.boxplot(data=data["Precipitation"])
# plt.show()
# print(data.duplicated().sum())

# t = data[["City", "AQI"]].sort_values("AQI")
# print("最好",t.iloc[:5])
# sns.barplot(x="City", y="AQI", data=t.iloc[:5])
# plt.show()
# print("最差",t.iloc[-5:])
# print(data["Coastal"].value_counts())
# sns.countplot(x="Coastal", data=data)
# plt.show()
# sns.violinplot(x="Coastal", y="AQI", data=data, inner=None)
# sns.swarmplot(x="Coastal", y="AQI", data=data, color="g")
# plt.show()

# plt.figure(figsize=(15,15))
# sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap=plt.cm.RdYlGn)
# plt.show()

# sns.scatterplot(x="Longitude", y="Latitude", hue="AQI", palette=plt.cm.RdYlGn_r, data=data)
# plt.show()

# all = np.random.normal(loc=30, scale=50, size=10000)
# mean_arr = np.zeros(2000)
# for i in range(len(mean_arr)):
#     mean_arr[i] = np.random.choice(all, size=50, replace=False).mean()
# print(mean_arr.mean())
# sns.kdeplot(mean_arr, shade=True)
# plt.show()

stats.ttest_1samp(data["AQI"], 71)

mean = data["AQI"].mean()
std = data["AQI"].std()
print(mean - 1.96 * (std / np.sqrt(len(data))), mean + 1.96 * (std / np.sqrt(len(data))))