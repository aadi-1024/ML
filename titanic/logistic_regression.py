from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()

data = pd.read_csv('./train.csv')

data.loc[data["Sex"] == 'male', "Sex"] = 1
data.loc[data["Sex"] == 'female', "Sex"] = 0
data.loc[data["Age"].isnull(), "Age"] = 29.59342

# lr_age = LogisticRegression() #estimating age using regression todo
# lr_age.fit(data.loc[data["Age"].notna()], data.loc[data["Age"].notna(), "Age"])

data_X, data_y = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]], data["Survived"]

# data_X = data_X.dropna(axis=0)

train_X, val_X, train_y, val_y = train_test_split(data_X, data_y)

scaler.fit_transform(train_X, train_X)
scaler.fit_transform(val_X, val_X)

lr_model = LogisticRegression(penalty='l2')
lr_model.fit(train_X, train_y)

# pred = lr_model.predict(val_X)
# print(mean_absolute_error(val_y, pred))

test = pd.read_csv('./test.csv')
pas = test["PassengerId"]

test.loc[test["Sex"] == 'male', "Sex"] = 1
test.loc[test["Sex"] == 'female', "Sex"] = 0

test.loc[test["Age"].isnull(), "Age"] = 30.27259
test.loc[test["Fare"].isnull(), "Fare"] = 35.627188

test_X = test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

scaler.fit_transform(test_X, test_X)

# print(test.describe())
pred = lr_model.predict(test_X)
# print(train_X.describe())

pd.concat([pas, pd.DataFrame(pred, columns=["Survived"])], ignore_index=True, axis=1).to_csv('final.csv', index=False)