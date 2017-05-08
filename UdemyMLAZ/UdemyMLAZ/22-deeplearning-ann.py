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