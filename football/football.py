import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set(style="darkgrid", font="SimHei", font_scale=1.5, rc={"axes.unicode_minus":False})

columns = ["Name", "Age", "Nationality", "Overall", "Potential", "Club", "Value", "Wage","Preferred Foot",
    "Position", "Jersey Number", "Joined", "Height", "Weight", "Crossing", "Finishing",
    "HeadingAccuracy", "ShortPassing", "Volleys", "Dribbling", "Curve", "FKAccuracy","LongPassing",
    "BallControl", "Acceleration", "SprintSpeed", "Agility", "Reactions", "Balance","ShotPower",
    "Jumping", "Stamina", "Strength", "LongShots", "Aggression", "Interceptions","Positioning", "Vision",
    "Penalties", "Composure", "Marking", "StandingTackle", "SlidingTackle", "GKDiving","GKHandling",
    "GKKicking", "GKPositioning", "GKReflexes", "Release Clause"]
file = pd.read_csv('data.csv', usecols=columns)
pd.set_option("max_columns",100)
# # print(file.isnull().sum(axis=1).value_counts())
file.dropna(axis=0, inplace=True)
# print(file.isnull().sum())
# print(file.describe())
# sns.boxplot(data=file[["Age", "Overall"]])
# plt.show()
# print(file.duplicated().sum())
# print(file[["Height", "Weight"]][0:5])

# def trans_height(height):
#     h = height.split("'")
#     return int(h[0]) * 30.48 + int(h[1]) * 2.54

# def trans_weight(weight):
#     w = int(weight.replace("lbs",""))
#     return w * 0.45

# file["Height"] = file["Height"].apply(trans_height)
# file["Weight"] = file["Weight"].apply(trans_weight)

# print(file[["Height", "Weight"]][0:5])

# fig, axes = plt.subplots(1, 2)
# fig.set_size_inches((18, 5))
# sns.distplot(file[["Height"]], bins=50, ax=axes[0], color="g")
# sns.distplot(file[["Weight"]], bins=50, ax=axes[1])
# plt.show()
# plt.savefig('height-weight.png')

# number = file["Preferred Foot"].value_counts()
# print(number)
# sns.countplot(x="Preferred Foot", data=file)
# plt.show()

# print(file.groupby("Preferred Foot")["Overall"].mean())
# sns.barplot(x="Preferred Foot", y="Overall", data=file)
# plt.show()

# t = file.groupby(["Preferred Foot", "Position"]).size()
# t = t.unstack()
# # print(t)
# t[t<50] = np.NaN
# t.dropna(axis=1, inplace=True)
# # print(t)

# t2 = file[file["Position"].isin(t.columns)]
# plt.figure(figsize=(18, 10))
# sns.barplot(x="Position", y="Overall", hue="Preferred Foot", hue_order=["Left", "Right"], data=t2)
# plt.show()

# g = file.groupby("Club")
# r = g["Overall"].agg(["mean", "count"])
# r = r[r["count"]>20]
# r = r.sort_values("mean", ascending=False).head(10)
# print(r)

# g = file.groupby("Nationality")
# r = g["Overall"].agg(["mean", "count"])
# r = r[r["count"]>50]
# r = r.sort_values("mean", ascending=False).head(10)
# # print(r)
# r.plot(kind="bar")
# plt.show()

# t = pd.to_datetime(file["Joined"])
# t = t.astype(np.str)

# join_year = t.apply(lambda item: int(item.split("-")[0]))
# over_five_year = (2018 - join_year) >= 5
# t2 = file[over_five_year]
# t2 = t2["Club"].value_counts()
# # print(t2)
# t2.iloc[:15].plot(kind="bar")
# plt.show()

# file2 = pd.read_csv("wc2018-players.csv",engine='python')
# print(file2.head())

# t = file2["Birth Date"].str.split(".", expand=True)
# t[0].value_counts().plot(kind="bar")
# t[1].value_counts().plot(kind="bar")
# t[2].value_counts().sort_index().plot(kind="bar")
# plt.show()

# g = file.groupby(["Jersey Number", "Position"])
# t = g.size()
# # print(t)
# t = t[t >= 100]
# t.plot(kind="bar")
# plt.show()

# def to_numeric(item):
#     # print(type(item), item)
#     k = 1
#     if item[-1] == "M":
#         k = 1000
#     value = float(re.sub('[^0-9.]','',item)) * k
#     return value

# file["Value"] = file["Value"].apply(to_numeric)
# file["Wage"] = file["Wage"].apply(to_numeric)
# file["Release Clause"] = file["Release Clause"].apply(to_numeric)
# # print(file.head())
# fig, ax = plt.subplots(1,3)
# fig.set_size_inches(18,5)

# sns.scatterplot(x="Value", y="Wage", data=file, ax=ax[0])
# sns.scatterplot(x="Value", y="Release Clause", data=file, ax=ax[1])
# sns.scatterplot(x="Wage", y="Release Clause", data=file, ax=ax[2])
# plt.show()

# print(file.corr())
# plt.figure(figsize=(40,40))
# sns.heatmap(file.corr(), annot=True, fmt=".2f", cmap=plt.cm.Greens)
# plt.savefig("corr.png", dpi=100, bbox_inches="tight")

# g = file.groupby("Position")
# f = g["GKDiving"].mean().sort_values(ascending=False)
# print(f.head())
# plt.figure(figsize=(15,5))
# sns.barplot(x="Position", y="GKDiving", data=file)
# plt.show()

# sns.scatterplot(x='Age', y='Overall', data=file)
# plt.show()

min_, max_ = file['Age'].min() -0.5, file['Age'].max()
t = pd.cut(file['Age'], bins=[min_, 20, 30, 40, max_], labels=['弱冠之年', 
    '而立之年','不惑之年','知天命'])
t = pd.concat((t, file['Overall']), axis=1)
g = t.groupby('Age')
# print(g['Overall'].mean())
sns.lineplot( x="Age", y="Overall",ci='sd', data=t)
plt.show()