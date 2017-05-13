dataset <- read.csv("..\\DeepLearning_ANN\\Churn_Modelling.csv", header = TRUE)


library(caTools)

#filter & clean from the dataset
dataset <- dataset[4:14]

#we don't need to encode target feature as a factor

#we need to encode geography & gender to factors and set them to numeric
#this is because the DL package requires numeri factors

dataset$Geography <- as.numeric(factor(dataset$Geography, levels = c('France', 'Spain', 'Germany'), labels = c(1, 2, 3)))
dataset$Gender <- as.numeric(factor(dataset$Gender, levels = c('Female', 'Male'), labels = c(0, 1)))

#split dataset
set.seed(123)
split <- sample.split(dataset$Exited, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

#feature scaling for everything except the target feature
training_set[-11] <- scale(training_set[-11])
test_set[-11] <- scale(test_set[-11])

#build the ANN
#install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)

#hidden is c(# of neurons, # of neurons in second hidden layer)
classifier <- h2o.deeplearning(
                            y = 'Exited',
                            training_frame = as.h2o(training_set),
                            activation = 'Rectifier',
                            hidden = c(6, 6),
                            epochs = 100,
                            train_samples_per_iteration = -2 )

prob_pred <- h2o.predict(classifier, newdata = as.h2o(test_set[-11]))
probdf <- as.data.frame(prob_pred)
y_pred <- as.vector(prob_pred > 0.5)
y_test <- test_set[, 11]

#building confusion matrix
cm <- table(y_test, y_pred)
score <- (cm[1, 1] + cm[2, 2]) / nrow(test_set)

#don't forget to shutdown h2o
h2o.shutdown(prompt = FALSE)
