#install.packages('arules')

library(arules)
dataset = read.csv("..\\AssociationRulesLearning_Eclat\\Market_Basket_Optimisation.csv", header = FALSE)

#use read transactions to build a sparse matrix
dataset = read.transactions("..\\AssociationRulesLearning_Eclat\\Market_Basket_Optimisation.csv", sep = ',', rm.duplicates = TRUE)

summary(dataset)

#plot items frequency
itemFrequencyPlot(dataset, topN=10)

#training eclat
rules = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2))

#insepction of rules
inspect(sort(rules, by = 'support')[1:10])
