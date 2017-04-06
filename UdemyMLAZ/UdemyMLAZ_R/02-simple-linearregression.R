dataset = read.csv("..\\Simple_Linear_Regression\\Salary_Data.csv")


# split the data
library(caTools)
set.seed(seed = 123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#fit data using linear regression
regressor = lm(formula = Salary ~ YearsExperience, training_set)

#notice the stars indicating statistically significant models
summary(regressor)

# prepare predictions vector
y_pred = predict(regressor, newdata = test_set)

#let's plot data
#install.packages('ggplot2')
library(ggplot2)

#plot observation points on training set
ggplot() +
    geom_point(
        aes(x = training_set$YearsExperience, y = training_set$Salary),
        color = 'red') +
    geom_line(
        aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
        color = 'green') +
    ggtitle('Salary vs Experience of Training Set') +
    xlab('Years of Experience') +
    ylab('Salary')

#plot observation points on test set
ggplot() +
    geom_point(
        aes(x = test_set$YearsExperience, y = test_set$Salary),
        color = 'red') +
#you don't change the prediction on the training set
    geom_line(
        aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
        color = 'green') +
    ggtitle('Salary vs Experience of Training Set') +
    xlab('Years of Experience') +
    ylab('Salary')
