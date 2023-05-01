import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./train.csv')

data.loc[data["Sex"] == 'male', "Sex"] = 1
data.loc[data["Sex"] == 'female', "Sex"] = 0

data.loc[data["Age"].isnull(), "Age"] = 29.59342

data_X = data[["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
data_y = data["Survived"]

train_X, val_X, train_y, val_y = train_test_split(data_X, data_y, train_size=0.8)

scale = StandardScaler()
scale.fit_transform(train_X, train_X)

nb_model = GaussianNB()
nb_model.fit(train_X[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]], train_y)

# test = pd.read_csv('./test.csv')
# pas = test['PassengerId']

# test.loc[test["Sex"] == 'male', "Sex"] = 1
# test.loc[test["Sex"] == 'female', "Sex"] = 0

# test.loc[test["Age"].isnull(), "Age"] = 30.27259
# test.loc[test["Fare"].isnull(), "Fare"] = 35.627188
# #replacing nan values with average values in data

# test_X = test[["Pclass", "Sex", "Age", "SibSp", "Fare"]] # only usable fields

# scale.fit_transform(test_X, test_X)

# # print(test.describe())
# pred = nb_model.predict(test_X)
# # print(train_X.describe())

# pd.concat([pas, pd.DataFrame(pred, columns=["Survived"])], ignore_index=True, axis=1).to_csv('final.csv', index=False)

scale.fit_transform(val_X, val_X)
pred = nb_model.predict(val_X[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]])

print(mean_absolute_error(val_y, pred))
# apparently scaling dataset improves accuracy