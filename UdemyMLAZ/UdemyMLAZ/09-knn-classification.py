import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
np.set_printoptions(threshold=np.nan)

# Importing the dataset
try:
    dataset = pd.read_csv('..\\LogisticRegression\\Social_Network_Ads.csv')
except:
    dataset = pd.read_csv('UdemyMLAZ\\LogisticRegression\\Social_Network_Ads.csv')

dataset

#include only Age & salary as features (try adding gender later)
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

#split data
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# apply feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#fitting logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

