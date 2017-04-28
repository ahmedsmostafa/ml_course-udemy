import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
np.set_printoptions(threshold=np.nan)

# Importing the dataset
try:
    dataset = pd.read_csv('..\\AssociationRulesLearning_Apriori\\Market_Basket_Optimisation.csv', heder=None)
except:
    dataset = pd.read_csv('UdemyMLAZ\\AssociationRulesLearning_Apriori\\Market_Basket_Optimisation.csv', header=None)

dataset

#extract features & outcomes from data