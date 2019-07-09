import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv('E:/dataAnalysis/Titanic/train.csv', engine="python")
# print(data_train.shape)
pd.set_option("max_columns", 20)
# print(data_train.head())
# print(data_train.info())
# print(data_train.isnull().sum(axis=1).value_counts())
# print(data_train.describe())
# print(data_train.duplicated().sum())

# fig = plt.figure()
# fig.set(alpha=0.2)

# plt.subplot2grid((2,3),(0,0))
# data_train.Survived.value_counts().plot(kind='bar')
# plt.ylabel(u'人数')
# plt.title(u'获救情况 (1为获救)')

# plt.subplot2grid((2,3),(0,1))
# data_train.Pclass.value_counts().plot(kind='bar')
# plt.ylabel(u'人数')
# plt.title(u'乘客等级分布')

# plt.subplot2grid((2,3),(0,2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel(u'年龄')
# plt.grid(b=True, which='major',axis='y')
# plt.title(u'按年龄看获救分布（1为获救）')

# plt.subplot2grid((2,3),(1,0),colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u'年龄')
# plt.ylabel(u'密度')
# plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')


# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u'各登船口岸上船人数')
# plt.ylabel(u'人数')
# plt.show()


# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u"获救":Survived_1, u"未获救":Survived_0})
# # print(df.stack())
# df.plot(kind="bar", stacked=True)
# plt.title(u"各乘客等级的获救情况")
# plt.xlabel(u"乘客等级")
# plt.ylabel(u"人数")
# plt.legend(loc="best")
# plt.show()

# Survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u"获救":Survived_1, u"未获救":Survived_0})
# df.plot(kind="bar", stacked=True)
# plt.title(u"男女乘客的获救情况")
# plt.xlabel(u"乘客性别")
# plt.ylabel(u"人数")
# plt.legend(loc="best")
# plt.show()

# ax1 = fig.add_subplot(141)
# data_train.Survived[data_train.Sex == "female"][data_train.Pclass != 3].value_counts().plot(kind="bar", color="orange")
# ax1.legend([u'女性/高级舱'], loc="best")
# ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)

# ax2 = fig.add_subplot(142)
# data_train.Survived[data_train.Sex == "female"][data_train.Pclass == 3].value_counts().plot(kind="bar", color="pink")
# ax2.legend([u'女性/低级舱'], loc="best")
# ax2.set_xticklabels([u"获救", u"未获救"], rotation=0)

# ax3= fig.add_subplot(143)
# data_train.Survived[data_train.Sex == "male"][data_train.Pclass != 3].value_counts().plot(kind="bar", color="blue")
# ax3.legend([u'男性/高级舱'], loc="best")
# ax3.set_xticklabels([u"获救", u"未获救"], rotation=0)

# ax4 = fig.add_subplot(144)
# data_train.Survived[data_train.Sex == "male"][data_train.Pclass == 3].value_counts().plot(kind="bar", color="green")
# ax4.legend([u'男性/低级舱'], loc="best")
# ax4.set_xticklabels([u"获救", u"未获救"], rotation=0)

# plt.title(u"根据舱等级和性别的获救情况")
# plt.show()

# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u"获救":Survived_1, u"未获救":Survived_0})
# df.plot(kind="bar", stacked=True)
# plt.title(u"各港口的获救情况")
# plt.xlabel(u"港口")
# plt.ylabel(u"人数")
# plt.legend(loc="best")
# plt.show()

# g = data_train.groupby(['SibSp', 'Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)

# p = data_train.groupby(['Parch', 'Survived'])
# df = pd.DataFrame(p.count()['PassengerId'])
# print(df)

# print(data_train.Cabin.value_counts())

# Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# df = DataFrame({u"有":Survived_cabin, u"无":Survived_nocabin}).transpose()
# # print(df)
# df.plot(kind="bar", stacked=True)
# plt.title(u"按Cabin看获救情况")
# plt.xlabel(u"Cabin有无")
# plt.ylabel(u"人数")
# plt.show()

from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    y = known_age[:, 0]
    X = known_age[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=500, n_jobs=-1)
    rfr.fit(X, y)

    predictedAges = rfr.predict(unknown_age[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
# print(data_train.head())
# print(data_train.info())

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Cabin', 'Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)
# print(df.head())
# print(df.info())

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
# age_scale_param = scaler.fit(df[['Age']])
df['Age_scaled'] = scaler.fit_transform(df[['Age']])
# fare_scale_param = scaler.fit(df[['Fare']])
df['Fare_scaled'] = scaler.fit_transform(df[['Fare']])
# print(df.head())

from sklearn import linear_model

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

y = train_np[:, 0]
X = train_np[:, 1:]

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

# print(clf)

data_test = pd.read_csv('E:/dataAnalysis/Titanic/test.csv')
# print(data_test.isnull().sum())
# print(data_test.shape)
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
tem_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tem_df[data_test.Age.isnull()].values
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull(), 'Age')] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Cabin', 'Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age']])
df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Fare']])

test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
# print(type(predictions[0]))
# print(type(data_test['PassengerId'].values))
# result = pd.DataFrame({'PassengerId': data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
# result.to_csv("E:/dataAnalysis/Titanic/logistic_regression_predictions.csv", index=False)

# print(pd.DataFrame({'columns': list(train_df.columns)[1:], 'coef': list(clf.coef_.T)}))

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import learning_curve 

# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# all_data = df.filter(regex='Survived|Age_.*|SibSp|Prach|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# X = all_data.values[:, 1:]
# y = all_data.values[:, 0]
# print(cross_val_score(clf, X, y, cv=5))

# split_train, split_cv = train_test_split(df, test_size=0.3, random_state=0)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Prach|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# clf.fit(train_df.values[:, 1:], train_df.values[:, 0])
# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Prach|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(cv_df.values[:, 1:])
# origin_data_train = pd.read_csv('E:/dataAnalysis/Titanic/train.csv', engine="python")
# bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:, 0]]['PassengerId'].values)]
# print(bad_cases)

# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
#                         train_sizes = np.linspace(.05, 1., 20), verbose=0, plot=True):
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)

#     if plot:
#         plt.figure()
#         plt.title(title)
#         if ylim is not None:
#             plt.ylim(*ylim)
#         plt.xlabel(u'训练样本数')
#         plt.ylabel(u'得分')
#         # plt.gca().invert_yaxis()
#         plt.grid()

#         plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
#                         alpha=0.1, color='b')
#         plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
#                         alpha=0.1, color='r')
#         plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label=u'训练集上得分')
#         plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label=u'交叉验证集上得分')

#         plt.legend(loc='best')

#         plt.draw()
#         plt.show()
#         # plt.gca().invert_yaxis()

#     midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
#     diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
#     print(midpoint, diff)
# plot_learning_curve(clf, u'学习曲线', X, y, ylim = (0.75,0.9))

from sklearn.ensemble import BaggingRegressor
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.values

y = train_np[:, 0]
X = train_np[:, 1:]

clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)

clf.fit(X, y)
