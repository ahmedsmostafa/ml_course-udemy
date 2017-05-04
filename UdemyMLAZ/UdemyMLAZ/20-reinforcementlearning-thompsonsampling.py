import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
np.set_printoptions(threshold=np.nan)

# Importing the dataset
import sys

try:
    dataset = pd.read_csv('..\\ReinforcementLearning_ThompsonSampling\\Ads_CTR_Optimisation.csv')
    #there is no apriori implementation in python?
    #using the file given in our examples apyori.py
    #sys.path.insert(0, '..\\AssociationRulesLearning_Apriori\\')
except:
    #there is no apriori implementation in python?
    #using the file given in our examples apyori.py
    dataset = pd.read_csv('UdemyMLAZ\\ReinforcementLearning_ThompsonSampling\\Ads_CTR_Optimisation.csv')
    #sys.path.insert(0, 'UdemyMLAZ\\AssociationRulesLearning_Apriori\\')

dataset

#implement thompson sampling step by step
import random
"""
step 1: declare 2 variables, 
for each round n:
Ni1(n): the count of times the ad i was rewarded 1
Ni0(n): the count of times the ad i was rewarded 0
we do 10,000 rounds to the size of dataset
"""
N = len(dataset)
d = len(dataset.columns)
ads_selected = []
total_reward = 0

count_of_rewards_1 = [0] * d
count_of_rewards_0 = [0] * d
for n in range(0,N):
    """
    step2: for each ad i, take a random draw from the distrbution below:
    theta(n) = beta(Ni1(n)+1, Ni0(n)+1)
    """
    max_random = 0
    ad = 0
    for i in range(0,d):
        random_beta = random.betavariate(count_of_rewards_1[i] + 1, count_of_rewards_0[i] + 1)
        
        #calculate max random among all ads
        #keep track of the index of the ad
        #choose the maximum random variable
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    """
    we update the selected ads, calculate reward, update it for every selected ad
    """
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if reward == 1:
        count_of_rewards_1[ad] += 1
    else:
        count_of_rewards_0[ad] += 1
    total_reward += reward

"""
step3: we select the ad i that has the higest theta(n) 
"""
best_ad = count_of_rewards_1.index(max(count_of_rewards_1))
print("the best ad is Ad %d with value %d" % (best_ad, count_of_rewards_1[best_ad]))

#plot ads and access frequency
plt.hist(ads_selected)
plt.title("Ads access frequency")
plt.xlabel("Ads")
plt.ylabel("Access frequency")
plt.show()