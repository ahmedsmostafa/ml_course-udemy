# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.nan)

# Importing the dataset
dataset = pd.read_csv('..\\RandomForest_Regression\\Position_Salaries.csv')
dataset

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

'''
# STEP 1: pick at random K data points from the training set
# STEP 2: build a decision tree associated to the K data points
# STEP 3: Choose the number of Ntree of trees you want to build, and repeat STEP 1 & STEP 2
# STEP 4: For a new data point, make each one of your Ntree trees predict the value of Y for 
        the data point in question, and assign the new data point the average across 
        all of the predicted Y's
'''
# Fitting RF to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

#prediction
y_pred = regressor.predict(6.5)
y_pred[0]

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()