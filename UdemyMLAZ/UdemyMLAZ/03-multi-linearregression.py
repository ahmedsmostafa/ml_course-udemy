
'''
# y = b0 (for profit) + b1.x1 (for r&d) + b2.x2 (for admin) + b3.x3 (marketing) + b4.D1 (state is categorical)
# D1: to achieve correct regression, you need to convert the state column to categories of 0/1 for each value in it.
# you should use only 1 dummy variable (for example you choose newyork only and use 0/1)
# always omit one dummy variable (if you 3 use 2, if you have 100 use 99)
'''

'''
#backward elimination
step1: set a significance level to stay in the model; say SL=0.05
step2: fit the model with all possible predictors
step3: consider the predictor with the highest p-value, if p>SL, go to step4, otherwise end
step4: remove the predictor
step5: go to step2
end
'''

'''
#forward selection
step1: select significance level
step2: fit all simple regression models, select the one with the lowest p-value
step3: keep this variable and fit all possible models with one extra predictor added t the one(s) you already have
step4: consider the predictor with the lowest p-value, if p<SL go to step3, otherwise end
end
'''

'''
#bidirectional elimination
step1: select significance level to enter & to stay
step2: perform forward selection (new vars must have p < SLEnter to enter)
step3: perform backward elimination (old vars must have p < SLStay to stay)
step4: keep repeating 2-3 until no vars enter/exist
'''

'''
#all-possible-models
step1: select a crtierion of goodness of fit
step2: construct 2^n-1 total combination of models for all features
step3 seelct the one with the best critierion
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

np.set_printoptions(threshold=np.nan)

def applyBackwardElimination(X,y):
    # NOW: building optimal model using backward elimination, 
    # the lower Pvalue the more significant it is
    # our SL value is 0.05
    import statsmodels.formula.api as sm
    # add intercept column of one for X0 (remember?)
    X = np.append(arr = np.ones((len(X),1)).astype(int), values = X, axis = 1)

    # build the backward elimination by removing the features that are not statistically significant
    indices_list=list(range(X.shape[1]))
    X_opt = X[:,indices_list]

    significance_level=0.05

    while(True):
        #step2: build regressor to fit all data as backward elimination states
        regressor_ols = sm.OLS(endog=y, exog=X_opt)
        results = regressor_ols.fit()
        # step3: look for the predictor
        # results.summary()
        #get the column index that has the maximum pvalue, subtract 1 because it returns index of X0
        maxpvalueindex = np.where(results.pvalues == max(results.pvalues))[0][0]
        print('maxPvalueIndex=%d' % maxpvalueindex)
        maxpvalue = results.pvalues[maxpvalueindex]
        print('maxPvalue=%d' % maxpvalue)
        
        if maxpvalue <= significance_level:
            return results
        
        #remove that max pvalue index from the indices list
        del indices_list[maxpvalueindex]
        X_opt = X[:,indices_list]


current_dir = os.getcwd()
dataset = pd.read_csv("..\\Multiple_Linear_Regression\\50_Startups.csv")
dataset

#matrix of our inputs
X = dataset.iloc[:,:-1].values

#facts vector
y = dataset.iloc[:,-1].values

'''
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
'''

    # categorize the string COUNTRIES column to numerical categories using Scikit learn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])

    # convert numerical categories to independnet dummy variables
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

    # Avoiding the dummy variable trap
X = X[:,1:]


    # NOW the data is cleansed, you should SPLIT the data to training & test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

'''
    # Not needed as MLR will take care of feature scaling
    # Once data is cleansed, categorized.. you need to apply feature scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)
'''

    # apply linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

    # apply prediction
y_pred = regressor.predict(X_test)

results = applyBackwardElimination(X,y)

print(results.summary())