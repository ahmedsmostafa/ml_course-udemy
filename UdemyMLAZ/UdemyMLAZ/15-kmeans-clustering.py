import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
np.set_printoptions(threshold=np.nan)

# Importing the dataset
try:
    dataset = pd.read_csv('..\\KMeans_Clustering\\Mall_Customers.csv')
except:
    dataset = pd.read_csv('UdemyMLAZ\\KMeans_Clustering\\Mall_Customers.csv')

dataset

#extract features & outcomes from data
X = dataset.iloc[:, [-2, -1]].values

#there is no y because we have no predictions/outcomes

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

# calculate the within cluster sum of square and plot the 10 iterations
# wcss = inertia

wcss = []
silhouette_scores = []
for i in range(1,11):
    # we will fit the kmeans to X using i
    kmeans = KMeans(n_clusters = i, init= 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    # calculate wcss and append it to wcss list
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("The elbow method")
plt.xlabel("Clusters #")
plt.ylabel("WCSS")
plt.show()

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
ykmeans = kmeans.fit_predict(X)

#visualize clusters
plt.title('Clusters of customers')
plt.xlabel('Annual Income in 1000')
plt.ylabel('Spending Score')
plt.scatter(X[ykmeans == 0,0], X[ykmeans == 0, 1], s = 100, c = 'red', label='Careful')
plt.scatter(X[ykmeans == 1,0], X[ykmeans == 1, 1], s = 100, c = 'blue', label='Standard')
plt.scatter(X[ykmeans == 2,0], X[ykmeans == 2, 1], s = 100, c = 'green', label='Platinum')
plt.scatter(X[ykmeans == 3,0], X[ykmeans == 3, 1], s = 100, c = 'cyan', label='Careless')
plt.scatter(X[ykmeans == 4,0], X[ykmeans == 4, 1], s = 100, c = 'magenta', label='Senisble')
#plt.show()
#plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.legend()
plt.show()
