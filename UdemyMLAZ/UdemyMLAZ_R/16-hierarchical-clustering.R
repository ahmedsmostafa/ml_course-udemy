dataset = read.csv("..\\Hierarchical_Clustering\\Mall_Customers.csv")
dataset

#select only relevant data
X <- dataset[4:5]
X

#create dendrogram
dendrogram = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendrogram, main='Dendrogram', xlab='Customers', ylab='Euclidean Distance')

#fitting hierarchical clustering to dataset
hc = hclust(dist(X, method = 'euclidean'), method = 'ward.D')
yhc = cutree(hc, 5)

#visualize the clusters
library(cluster)
clusplot(X,
    yhc,
    lines = 0,
    shade = TRUE,
    color = TRUE,
    labels = 2,
    plotchar = TRUE,
    span = TRUE,
    main = paste('Clusters of Customers'),
    xlab = 'Annual Income',
    ylab = 'Spending Score')