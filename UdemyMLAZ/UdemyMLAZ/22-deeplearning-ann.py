#installing theano
#pip install --upgrade --no-deps git+git://github/Theano/Theano.git

#installing Tensorflow
#install from website

#installing keras
#pip install --upgrade keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
np.set_printoptions(threshold=np.nan)

# Importing the dataset
import sys

try:
    dataset = pd.read_csv('..\\DeepLearning_ANN\\Churn_Modelling.csv')
    #there is no apriori implementation in python?
    #using the file given in our examples apyori.py
    #sys.path.insert(0, '..\\AssociationRulesLearning_Apriori\\')
except:
    #there is no apriori implementation in python?
    #using the file given in our examples apyori.py
    dataset = pd.read_csv('UdemyMLAZ\\DeepLearning_ANN\\Churn_Modelling.csv')
    #sys.path.insert(0, 'UdemyMLAZ\\AssociationRulesLearning_Apriori\\')

dataset

#extract features, and dependant variables
X = dataset[dataset.columns[3:13]].values
y = dataset["Exited"].values

#encode categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()
X[:,1] = labelencoder_country.fit_transform(X[:,1])

labelencoder_gender = LabelEncoder()
X[:,2] = labelencoder_gender.fit_transform(X[:,2])

#create dummy variable for country to solve the ordinal issue of its categorical values
#passing 1 to indicate the index of which feature we want to operate on
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#remove first column to avoid the dummy trap
X = X[:,1:]

#split data set into training & test
#from sklearn.cross_validation import train_test_split
#instead of using sklearn.cross_validation use sklearn.model_selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#apply feature scaling to easy the heavy computations
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#building ANN
import keras
#sequentia module used to initialize NN
from keras.models import Sequential

# module used to build layers of NN
from keras.layers import Dense

#initializing the ANN
#it is done either by defining the sequence of layers, or defining the graph
classifier = Sequential()

#define first hidden layer
#will have 6 nodes, based on 11 features + 1 output divided by 2
#activation function will be rectifier (relu) or sigmoid
#RECOMMENDED:
#  rectifier function is for hidden layer, 
#  sigmoid is for output
#input_dim will be the size of features

#this line below uses old keras API
#classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim = 11))

#this line means we have a hidden layer of 6 nodes, and 11 input representing all features 
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim = 11))

#add the second hidden layer
#input_dim isn't needed because it will get its input from the layer created before
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

#add the output layer, only 1 node/unit, use sigmoid activation
#if you have multiple categories you choose soft max
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#fitting ANN to training set
#batch size is after how much you want to update the weights
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)