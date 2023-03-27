import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# get correlated columns
columns = ['Tonnage', 'passengers', 'length', 'cabins', 'crew']

df = pd.read_csv('../dataset/dataset.csv', sep=',', usecols=columns)

df.isnull().sum()

# remove rows that contain missing values
df = df.dropna(axis=0)
df.isnull().sum()

# dataset
X = df[columns]

target = 'crew'
features = df.columns[df.columns != target]

# input data
X = df[features].values
# output data
y = df[target].values

# spit dataset in train and test in order to evaluate the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# linear regression model
slr = LinearRegression()

# train the model
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# evaluate the model
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'MSE train: {mse_train:.2f}')
print(f'MSE test: {mse_test:.2f}')

r2_train = r2_score(y_train, y_train_pred)
r2_test =r2_score(y_test, y_test_pred)
print(f'R^2 train: {r2_train:.2f}')
print(f'R^2 test: {r2_test:.2f}')

# results: R2 train and test > 90% 