#!/usr/bin/env python


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score


from sklearn.datasets import fetch_california_housing
housing_data = fetch_california_housing()


df = pd.DataFrame(housing_data.data)
df.columns = housing_data.feature_names
df.head()


df['MedHouseVal'] = housing_data.target
df.head()


df.info()


df.shape


df.isnull().sum()

df.describe()



print(housing_data)




g = sns.relplot(data=df, x="MedInc", y="MedHouseVal")
g.ax.axline(xy1=(10, 2), slope=.2, color="b", dashes=(5, 2))


df.hist(figsize=(150,100), bins=100, edgecolor="black")
plt.show()


plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="Longitude", y="Latitude", hue="MedHouseVal", size="MedHouseVal",
               palette="viridis", alpha=0.5)
plt.legend(title="MedHouseVal", loc="upper right")
plt.title("Median housing value depending on \n their spatial location")


plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="Longitude", y="Latitude",
               palette="viridis", alpha=0.5)
plt.title("Same plot without setting the hue and size para")


df.corr()

plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation between the attributes")
plt.show()

df.corr()['MedHouseVal'].sort_values()


sns.scatterplot(x=df['MedInc'], y=df['MedHouseVal'])

sns.scatterplot(x=df['AveRooms'], y=df['MedHouseVal'])

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']


#SPLITING THE DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

scaler = StandardScaler()
X_train  = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
linreg_pred = linreg.predict(X_test)

linreg_mae = mean_absolute_error(y_test, linreg_pred)
linreg_mse = mean_squared_error(y_test, linreg_pred)
linreg_r2 = r2_score(y_test, linreg_pred)
print("MAE of the linear regression model is:", linreg_mae)
print("MSE of the linear regression model is:", linreg_mse)
print("R2 score of the linear regression model is:", linreg_r2)


# ### Decision Tree
dtree = DecisionTreeRegressor()
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)

dtree_mae = mean_absolute_error(y_test, dtree_pred)
dtree_mse = mean_squared_error(y_test, dtree_pred)
dtree_r2 = r2_score(y_test, dtree_pred)
print("MAE of the decision tree model is:", dtree_mae)
print("MSE of the decision tree model is:", dtree_mse)
print("R2 score of the decision tree model is:", dtree_r2)


# ### Random forest
rforest = RandomForestRegressor()
rforest.fit(X_train, y_train)
rforest_pred = rforest.predict(X_test)

rforest_mae = mean_absolute_error(y_test, rforest_pred)
rforest_mse = mean_squared_error(y_test, rforest_pred)
rforest_r2 = r2_score(y_test, rforest_pred)
print("MAE of the random forest model is:", rforest_mae)
print("MSE of the random forest model is:", rforest_mse)
print("R2 score of the random forest model is:", rforest_r2)





#  HOUSE IN RURAL CALIFORNIA
# data = {'MedInc':7.325, 'HouseAge':30.0, 'AveRooms':5.984, 'AveBedrms':1.0238,
#         'Population':280, 'AveOccup':2.20,'Latitude':40.86, 'Longitude':-120.46}


#SAME EXACT HOUSE IN LA CENTRE
data = {'MedInc':1.5603, 'HouseAge':25.0, 'AveRooms':5.0, 'AveBedrms':1.13,
        'Population':845, 'AveOccup':2.56,'Latitude':39.50, 'Longitude':-121.09}
index = [0]
new_df = pd.DataFrame(data, index)

df.columns
value_pred = dtree.predict(new_df)
print("The median housing value for the new data is: ", value_pred)