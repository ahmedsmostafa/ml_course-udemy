# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
np.set_printoptions(threshold=np.nan)

# Importing the dataset
try:
    dataset = pd.read_csv('..\\SupportVector_Regression\\Position_Salaries.csv')
except:
    dataset = pd.read_csv('UdemyMLAZ\\SupportVector_Regression\\Position_Salaries.csv')
dataset

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling becase SVR doesn't scale features.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting the Regression Model to the dataset
# Create your regressor here
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

# Predicting a new result
#you can transform using 
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
y_pred
#you need to inverse using sc_y to get the actual scaled prediction
y_pred = sc_X.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
y_pred

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()