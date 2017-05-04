import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
np.set_printoptions(threshold=np.nan)

# Importing the dataset
import sys

try:
    dataset = pd.read_csv('..\\AssociationRulesLearning_Apriori\\Market_Basket_Optimisation.csv', header=None)
    #there is no apriori implementation in python?
    #using the file given in our examples apyori.py
    #sys.path.insert(0, '..\\AssociationRulesLearning_Apriori\\')
except:
    #there is no apriori implementation in python?
    #using the file given in our examples apyori.py
    dataset = pd.read_csv('UdemyMLAZ\\AssociationRulesLearning_Apriori\\Market_Basket_Optimisation.csv', header=None)
    #sys.path.insert(0, 'UdemyMLAZ\\AssociationRulesLearning_Apriori\\')

dataset

# convert dataset to a sparse matrix
#create empty vector/list
#use dataset.values to navigate through them by index
#surround the result by string to make sure nan is set to string 
#surround the whole line by a [] to make sure each row is a list itself
transactions = []
for i in range(0 , len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(0,len(dataset.columns))])

# import the local apriori implementation
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

result = list(rules)
result
