import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
np.set_printoptions(threshold=np.nan)

# Importing the dataset
try:
    dataset = pd.read_csv('..\\Hierarchical_Clustering\\Mall_Customers.csv')
except:
    dataset = pd.read_csv('UdemyMLAZ\\Hierarchical_Clustering\\Mall_Customers.csv')

dataset