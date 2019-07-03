import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

# print(stats.ttest_1samp(data["AQI"], 71))

# mean = data["AQI"].mean()
# std = data["AQI"].std()
# print(mean - 1.96 * (std / np.sqrt(len(data))), mean + 1.96 * (std / np.sqrt(len(data))))

# X = data.drop(['City', 'AQI'], axis=1)
# y = data['AQI']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# lr = LinearRegression()
# lr.fit(X_train, y_train)
# y_hat = lr.predict(X_test)

# print(lr.score(X_train, y_train))
# print(lr.score(X_test, y_test))

# plt.figure(figsize=(15, 5))
# plt.plot(y_test.values, "-r", label="真实值")
# plt.plot(y_hat, "-g", label="预测值")
# plt.legend()
# plt.title("线性回归预测结果")
# plt.show()

X = data.drop(['City', 'Coastal'], axis=1)
y = data['Coastal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
lr = LogisticRegression(C=0.0001)
lr.fit(X_train, y_train)
y_hat = lr.predict(X_test)
# print(lr.score(X_train, y_train))
# print(lr.score(X_test, y_test))

# plt.figure(figsize=(15, 5))
# plt.plot(y_test.values, marker="o",c="r", ms=8, ls="",label="真实值")
# plt.plot(y_hat, marker="x", c="g", ms=8, ls="", label="预测值")
# plt.legend()
# plt.title("逻辑回归预测结果")
# plt.show()

probability = lr.predict_proba(X_test)
print(probability[:10])
print(np.argmax(probability, axis=1))
index = np.arange(len(X_test))
pro_0 = probability[:, 0]
pro_1 = probability[:, 1]
tick_label = np.where(y_test == y_hat, "O", "X")
plt.figure(figsize=(15, 5))
plt.bar(index, height=pro_0, color="g", label="类别0概率值")
plt.bar(index, height=pro_1, color="r", bottom=pro_0, label="类别1概率值", tick_label=tick_label)
plt.legend(loc="best", bbox_to_anchor=(1, 1))
plt.xlabel("样本序号")
plt.ylabel("各个类别的概率")
plt.title("逻辑回归分类概率")
plt.show()