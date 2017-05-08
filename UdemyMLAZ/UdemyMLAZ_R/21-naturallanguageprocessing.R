dataset = read.delim("..\\NaturalLanguageProcessing\\Restaurant_Reviews.tsv",
    header = TRUE, quote = '', stringsAsFactors = FALSE)

#cleaning the text
#install.packages('tm')
dependentVariable_Liked = dataset$Liked

library(tm)
corpus = VCorpus(VectorSource(dataset$Review))
as.character(corpus[[1]])

corpus = tm_map(corpus, content_transformer(tolower))
cat('to lower: ', as.character(corpus[[1]]), '\n')

corpus = tm_map(corpus, removeNumbers)
cat('remove numbers: ', as.character(corpus[[1]]), '\n')

corpus = tm_map(corpus, removePunctuation)
cat('remove punctuation:', as.character(corpus[[1]]), '\n')

corpus = tm_map(corpus, removeWords, stopwords())
cat('remove stop words: ', as.character(corpus[[1]]), '\n')

corpus = tm_map(corpus, stemDocument)
cat('stem document: ', as.character(corpus[[1]]),'\n')

corpus = tm_map(corpus, stripWhitespace)
cat('stem document: ', as.character(corpus[[841]]), '\n')

#install.packages('SnowballC')
# build sparse matrix
library(SnowballC)
dtm = DocumentTermMatrix(corpus)
dtm

#minimize matrix
dtm = removeSparseTerms(dtm, 0.999)
dtm

# create X,y from the dataset, build dataframe
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dependentVariable_Liked

#run classification models
# 1.running random forest

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))
dataset

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting classifier to the Training set
# Create your classifier here
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)
# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])
y_pred
test_set$Liked

# Making the Confusion Matrix
# cm = table(test_set[, 3], y_pred)
cm = table(test_set$Liked, y_pred)
cm