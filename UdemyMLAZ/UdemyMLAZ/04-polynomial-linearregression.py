import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

np.set_printoptions(threshold=np.nan)
current_dir = os.getcwd()
dataset = pd.read_csv("..\\Polynomial_Regression\\Position_Salaries.csv")
dataset

#matrix of our inputs
X = dataset.iloc[:,1:-1].values
X
#facts vector
y = dataset.iloc[:,-1].values
y

# splitting isn't necessary because of the data size and objective
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
"""

# apply regression
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X, y)

# apply polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
#you need to fit linear regression to X_poly
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (linear regression of facts)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))

plt.scatter(X, y, color='green')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (polynomial)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#perform actual predictions now with linear model
lin_reg.predict(6.5)

# perform actual prediction now with polynomial model
lin_reg2.predict(poly_reg.fit_transform(6.5))