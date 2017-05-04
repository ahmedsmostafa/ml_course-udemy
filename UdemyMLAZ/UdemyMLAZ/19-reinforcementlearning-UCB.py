import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
np.set_printoptions(threshold=np.nan)

# Importing the dataset
import sys

try:
    dataset = pd.read_csv('..\\ReinforcementLearning_UpperConfidenceBound\\Ads_CTR_Optimisation.csv')
    #there is no apriori implementation in python?
    #using the file given in our examples apyori.py
    #sys.path.insert(0, '..\\AssociationRulesLearning_Apriori\\')
except:
    #there is no apriori implementation in python?
    #using the file given in our examples apyori.py
    dataset = pd.read_csv('UdemyMLAZ\\ReinforcementLearning_UpperConfidenceBound\\Ads_CTR_Optimisation.csv')
    #sys.path.insert(0, 'UdemyMLAZ\\AssociationRulesLearning_Apriori\\')

dataset

#implement ucb step by step
import math
"""
step 1: declare 2 variables, 
for each round n:
Ni(n): the count of the ad i was selected up to round n
Ri(n): the sum of rewards of the ad i up to round n
we do 10,000 rounds to the size of dataset
"""
N = len(dataset)
count_of_selection = [0] * len(dataset.columns)
sum_of_rewards = [0] * len(dataset.columns)
ads_selected = []
total_reward = 0
best_ad = 0
for n in range(0,N):
    """
    step2: for each ad i 
    compute the average reward
    compute the confidence interval
    """
    max_upperbound = 0
    ad = 0
    for i in range(0,len(dataset.columns)):
        if(count_of_selection[i] > 0):
            average_reward = sum_of_rewards[i] / count_of_selection[i]
            #use log(n+1) because n starts with zero and log(0) is not defined
            delta_i = math.sqrt(3/2 * math.log(n+1)/count_of_selection[i])
            upperbound = average_reward + delta_i
        else:
            upperbound = 1e400
        #calculate max upperbound among all ads
        #keep track of the index of the ad
        if upperbound > max_upperbound:
            max_upperbound = upperbound
            ad = i
    """
    we update the selected ads, calculate reward, update it for every selected ad
    """
    ads_selected.append(ad)
    count_of_selection[ad] += 1
    reward = dataset.values[n,ad]
    sum_of_rewards[ad] += reward
    total_reward += reward
    
    """
    step3: we select the ad i that has the maximum upperbound
    """
    best_ad = sum_of_rewards.index(max(sum_of_rewards))
    print("the best ad is Ad %d with value %d" % (best_ad, sum_of_rewards[best_ad]))

#plot ads and access frequency
plt.hist(ads_selected)
plt.title("Ads access frequency")
plt.xlabel("Ads")
plt.ylabel("Access frequency")
plt.show()