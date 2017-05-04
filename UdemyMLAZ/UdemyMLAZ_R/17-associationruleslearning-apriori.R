#install.packages('arules')

library(arules)
dataset = read.csv("..\\AssociationRulesLearning_Apriori\\Market_Basket_Optimisation.csv", header = FALSE)

#use read transactions to build a sparse matrix
dataset = read.transactions("..\\AssociationRulesLearning_Apriori\\Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = FALSE)

summary(dataset)

#plot items frequency
itemFrequencyPlot(dataset, topN=10)

#training apriori
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

#insepction of rules
inspect(sort(rules, by = 'lift')[1:10])
