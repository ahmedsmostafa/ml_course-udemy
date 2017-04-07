dataset = read.csv("..\\Multiple_Linear_Regression\\50_Startups.csv")

#encoding categorical data
dataset$State = factor(dataset$State, levels = c('New York', 'California', 'Florida'), labels = c(1, 2, 3))


# split the data
library(caTools)
set.seed(seed = 123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#fit data using linear regression
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = training_set)
# you can use the following if you want to use all other features
# regressor = lm(formula = Profit ~ ., data = training_set)

#notice the stars indicating statistically significant models & Pvalues
summary(regressor)

# prepare predictions vector
y_pred = predict(regressor, newdata = test_set)

# implement backward elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = dataset)
summary(regressor)

max(summary(regressor)$coef[, "Pr(>|t|)"])
match(max(summary(regressor)$coef[, "Pr(>|t|)"]), summary(regressor)$coef[, "Pr(>|t|)"])

min(summary(regressor)$coef[, "Pr(>|t|)"])
match(min(summary(regressor)$coef[, "Pr(>|t|)"]), summary(regressor)$coef[, "Pr(>|t|)"])

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend, data = dataset)
summary(regressor)

