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

#extract features & outcomes from data
X = dataset.iloc[:, [-2, -1]].values

#there is no y because we have no predictions/outcomes

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# fitting hierarchical clustering to the dataset
from sklearn.cluster.hierarchical import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
yhc = hc.fit_predict(X)

#this could be used to automate plotting?
yhc
len(list(set(yhc)))

#visualize clusters
plt.title('Clusters of customers')
plt.xlabel('Annual Income in 1000')
plt.ylabel('Spending Score')
plt.scatter(X[yhc == 0,0], X[yhc == 0, 1], s = 100, c = 'red', label='Careful')
plt.scatter(X[yhc == 1,0], X[yhc == 1, 1], s = 100, c = 'blue', label='Standard')
plt.scatter(X[yhc == 2,0], X[yhc == 2, 1], s = 100, c = 'green', label='Platinum')
plt.scatter(X[yhc == 3,0], X[yhc == 3, 1], s = 100, c = 'cyan', label='Careless')
plt.scatter(X[yhc == 4,0], X[yhc == 4, 1], s = 100, c = 'magenta', label='Senisble')
#plt.show()
#plot centroids
#plt.scatter(hc.cluster_centers_[:,0], hc.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.legend()
plt.show()
