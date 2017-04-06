# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as datareader

np.set_printoptions(threshold=np.nan)

DATASET_PATH = "D:\\OneDrive\\Courses - Labs\\ML_AZ\\Machine Learning A-Z\\Part 1 - Data Preprocessing"
datafile = DATASET_PATH + "\\" + "Data.csv"

dataset = pd.read_csv(datafile)

#matrix of our inputs
X = dataset.iloc[:,:-1].values

#facts vector
y = dataset.iloc[:,-1].values

# we have 2 ways to fill NaN values:
## 1- calculating the mean yourself and filling it in your dataset
age_mean = dataset["Age"].mean()
salary_mean = dataset["Salary"].mean()
dataset["Age"].fillna(age_mean, inplace=True)
dataset["Salary"].fillna(salary_mean, inplace=True)

## 2- using Scikit Learn Imputer object, fit it, transform it
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# categorize the data using Scikit learn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# NOW the data is cleansed, you should SPLIT the data to training & test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Once data is cleansed, categorized.. you need to apply feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print("X:")
print(X)

print("y:")
print(y)

print("X_train:")
print(X_train)

print("X_test")
print(X_test)
