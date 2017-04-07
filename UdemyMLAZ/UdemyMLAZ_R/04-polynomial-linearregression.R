dataset = read.csv("..\\Polynomial_Regression\\Position_Salaries.csv")

dataset = dataset[2:3]

#encoding categorical data
#dataset$State = factor(dataset$State, levels = c('New York', 'California', 'Florida'), labels = c(1, 2, 3))

lin_reg = lm(formula = Salary ~ ., data = dataset)
summary(lin_reg)

#now polynomial regression
dataset$Level2 = dataset$Level ^ 2
dataset$Level3 = dataset$Level ^ 3
dataset$Level4 = dataset$Level ^ 4
poly_reg = lm(formula = Salary ~ ., data = dataset)

summary(poly_reg)

# visualize
library(ggplot2)
ggplot() +
    geom_point(
        aes(x = dataset$Level, y = dataset$Salary),
        color = 'red') +
    geom_line(
        aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
        color = 'green') +
    geom_line(
        aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
        color = 'blue')
    ggtitle('Level vs Salary of Dataset') +
    xlab('Level') +
    ylab('Salary')

#prediction of single value
# Predicting a new result with Linear Regression
y_lin_pred = predict(lin_reg, data.frame(Level = 6.5))
y_poly_pred = predict(lin_reg, data.frame(Level = 6.5, Level2 = 6.5^2, Level3=6.5^3, Level4=6.5^4))

#higher resolution, increase X values
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_reg,
                                        newdata = data.frame(Level = x_grid,
                                                             Level2 = x_grid ^ 2,
                                                             Level3 = x_grid ^ 3,
                                                             Level4 = x_grid ^ 4))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')


