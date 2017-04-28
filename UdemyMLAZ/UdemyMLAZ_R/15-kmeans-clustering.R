dataset = read.csv("..\\KMeans_Clustering\\Mall_Customers.csv")
dataset

#select only relevant data
X <- dataset[4:5]
X

# create en elbow to determine optimal number of clusters
set.seed(6)
wcss <- vector()
for (i in 1:10)
    wcss[i] <- sum(kmeans(X, i)$withinss)
plot(1:10, wcss, type='b', main = paste('Clusters of clients'), xlab = '# of clusters', ylab = 'WCSS')

# Fitting kmeans to the data
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)

#visualize the clusters
library(cluster)
clusplot(X,
    kmeans$cluster,
    lines = 0,
    shade = TRUE,
    color = TRUE,
    labels = 2,
    plotchar = TRUE,
    span = TRUE,
    main = paste('Clusters of Customers'),
    xlab = 'Annual Income',
    ylab = 'Spending Score')