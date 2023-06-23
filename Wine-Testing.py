import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


rf = RandomForestClassifier(random_state=1)
lr = LogisticRegression(random_state=1)
gbm = GradientBoostingClassifier(n_estimators=10)
dtc = DecisionTreeClassifier(random_state=1)

df = pd.read_csv("C:/Users/admin/PycharmProjects/pythonProject1/winequalityN.csv")

print(df.describe(),"/n")
print(df.dtypes)

df = df.rename(columns={"free sulfur dioxide": "free so2"})
df = df.rename(columns={"total sulfur dioxide": "total so2"})

# Treating Missing Values
print(df.isnull().sum())

# Replacing with their Means
print(df["fixed acidity"].value_counts())
mean = df["fixed acidity"].mean()
df["fixed acidity"].fillna(mean, inplace = True)
print(df["fixed acidity"].isnull().sum())

mean2 = df["volatile acidity"].mean()
df["volatile acidity"].fillna(mean, inplace = True)
print(df["volatile acidity"].isnull().sum())

mean3 = df["citric acid"].mean()
df["citric acid"].fillna(mean,inplace=True)
print(df["citric acid"].isnull().sum())

mean4 = df["residual sugar"].mean()
df["residual sugar"].fillna(mean,inplace=True)
print(df["residual sugar"].isnull().sum())

mean5 = df["chlorides"].mean()
df["chlorides"].fillna(mean,inplace=True)
print(df["chlorides"].isnull().sum())

mean6 = df["pH"].mean()
df["pH"].fillna(mean,inplace=True)
print(df["pH"].isnull().sum())

mean7 = df["sulphates"].mean()
df["sulphates"].fillna(mean,inplace=True)
print(df["sulphates"].isnull().sum())

print(df.isnull().sum())
df = df.drop(columns=["density","pH","sulphates"],axis=1)
print(df.head())

'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest (score_func=chi2 , k="all")
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
ftscores = pd.concat([dfcolumns,dfscores], axis=1)
ftscores.columns = ["Specs","Scores"]
print(ftscores)
'''
# Graphical Representation
'''
plt.figure(figsize = (10,6))
sns.boxplot(data = df, palette = "Set2")
plt.show()


plt.figure(figsize = (6, 5))
sns.boxplot(data = df['total so2'], palette = "Set3")
plt.show()


sns.set_style(style="darkgrid")
sns.countplot(x = "quality",hue="type", data = df)
plt.show()

# Removing Outliers
Q1 = df["free so2"].quantile(0.25)
print(Q1)

Q3 = df["free so2"].quantile(0.75)
print(Q3)

IQR = Q3 - Q1
print(IQR)

lower_limit = Q1 - 1.5 *(IQR)
upper_limit = Q3 + 1.5 *(IQR)
print(lower_limit, upper_limit)

print(df["free so2"].shape)

df2 = df[(df["free so2"] > lower_limit) & (df["free so2"] < upper_limit)]

print(df2["free so2"].shape)

print(df.shape[0] - df2.shape[0])

plt.figure(figsize = (10, 10))
sns.boxplot(data = df2, palette = "Set2")
plt.show()
'''
Q2 = df["total so2"].quantile(0.25)
print(Q2)

Q4 = df["total so2"].quantile(0.75)
print(Q4)