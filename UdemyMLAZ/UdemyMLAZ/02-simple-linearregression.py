#simple linear regression model to process data in the respective folder.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# 1- apply data pre-processing
# Data Preprocessing Template

# 2- Importing the dataset

current_dir = os.getcwd()
try:
    dataset = pd.read_csv("..\\Simple_Linear_Regression\\Salary_Data.csv")
except:
    dataset = pd.read_csv("UdemyMLAZ\\Simple_Linear_Regression\\Salary_Data.csv")
dataset

# 3- Split X,y
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 4- Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# 5- Feature Scaling not needed as most libraries take care of that already
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# 6- Fitting simple linear regression to the training set
# Use Scikit-learn linear model package to fit LR to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 7- predicting the test results
y_pred = regressor.predict(X_test)

# 8- plot the training results in scatter chart
plt.scatter(X_train, y_train, color='red')
# now plot this versus the prediction of the training set itself
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title('Salary vs Experience (training set results)')
plt.xlabel('X - years of experience')
plt.ylabel('y - salary')
plt.show()

# 9- plot the test results in scatter chart 
plt.scatter(X_test, y_test, color='red')
#plot this versus the prediction on the training set itself
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title('Salary vs Experience (training set results)')
plt.xlabel('X - years of experience')
plt.ylabel('y - salary')
plt.show()