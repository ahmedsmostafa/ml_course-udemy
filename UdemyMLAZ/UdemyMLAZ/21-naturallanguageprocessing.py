import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
np.set_printoptions(threshold=np.nan)

# Importing the dataset
import sys

#creating a named tuple to return the results of ML algorithms
import collections
MLResult = collections.namedtuple("MLResult", ['accuracy', 'precision', 'recall', 'f1_score'])

def runNaiveBayesML(X: np.ndarray, y: np.ndarray):
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # Fitting classifier to the Training set
    # Create your classifier here, play with different kernels now
    from sklearn.naive_bayes import GaussianNB,BaseDiscreteNB,BaseNB,BernoulliNB,MultinomialNB
    classifier = GaussianNB()
    classifier.fit(X_train,y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred
    y_test

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support
    cm = confusion_matrix(y_test, y_pred)
    cm

    score = classifier.score(X_test, y_test)
    
    #calculating evaluation metrics
    acc = accuracy_score(y_test,y_pred)
    precisionrecall = precision_recall_fscore_support(y_test, y_pred, average = 'weighted')
    f1score = f1_score(y_test, y_pred, average='weighted')
    result = MLResult(accuracy=acc, precision=precisionrecall[0], recall=precisionrecall[1], f1_score=f1score)
    return result

def runDecisionTreeML(X: np.ndarray, y: np.ndarray):
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # Fitting classifier to the Training set
    # Create your classifier here, play with different kernels now
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(X_train,y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred
    y_test

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support
    cm = confusion_matrix(y_test, y_pred)
    cm

    score = classifier.score(X_test, y_test)
    
    #calculating evaluation metrics
    acc = accuracy_score(y_test,y_pred)
    precisionrecall = precision_recall_fscore_support(y_test, y_pred, average = 'weighted')
    f1score = f1_score(y_test, y_pred, average='weighted')
    result = MLResult(accuracy=acc, precision=precisionrecall[0], recall=precisionrecall[1], f1_score=f1score)
    return result

def runRandomForestML(X: np.ndarray, y: np.ndarray):
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Fitting classifier to the Training set
    # Create your classifier here, play with different kernels now
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred
    y_test
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support
    cm = confusion_matrix(y_test, y_pred)
    cm
    
    score = classifier.score(X_test, y_test)
    
    #calculating evaluation metrics
    acc = accuracy_score(y_test,y_pred)
    precisionrecall = precision_recall_fscore_support(y_test, y_pred, average = 'weighted')
    f1score = f1_score(y_test, y_pred, average='weighted')
    result = MLResult(accuracy=acc, precision=precisionrecall[0], recall=precisionrecall[1], f1_score=f1score)
    return result

try:
    dataset = pd.read_csv('..\\NaturalLanguageProcessing\\Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)
    #there is no apriori implementation in python?
    #using the file given in our examples apyori.py
    #sys.path.insert(0, '..\\AssociationRulesLearning_Apriori\\')
except:
    #there is no apriori implementation in python?
    #using the file given in our examples apyori.py
    dataset = pd.read_csv('UdemyMLAZ\\NaturalLanguageProcessing\\Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)
    #sys.path.insert(0, 'UdemyMLAZ\\AssociationRulesLearning_Apriori\\')

dataset

#cleaning data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = [] * len(dataset)
corpus_unique = set([])
for rawreview in dataset['Review']:
    #selecting only alphabets
    review = re.sub('[^a-zA-Z]', ' ', rawreview).strip()
    #review

    #making everything lowercase
    review = review.lower()
    #review

    review = review.split()
    #review

    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #review
    
    corpus.append(' '.join(review))
    corpus_unique.update(review)

# convert reviews list to a numpy array
#dataset['Review'] = np.array(reviews)

# creating the bag of words model to minimize number of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
# Z is for my own testing
Z = cv.fit_transform(corpus, y = dataset['Liked'].values).toarray()

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# the most common classification models for NLP is:
# Naive Bayes, Decision Tree Classification, Random Forest

#Using Naive Bayes
NB_metrics = runNaiveBayesML(X,y)
DT_metrics = runDecisionTreeML(X,y)
RF_metrics = runRandomForestML(X,y)

print("NB accuracy = {0:2.2f}, precision = {1:2.2f}, recall = {2:2.2f}, f1_score = {3:2.2f}".format(NB_metrics.accuracy, NB_metrics.precision, NB_metrics.recall, NB_metrics.f1_score))
print("DT accuracy = {0:2.2f}, precision = {1:2.2f}, recall = {2:2.2f}, f1_score = {3:2.2f}".format(DT_metrics.accuracy, DT_metrics.precision, DT_metrics.recall, DT_metrics.f1_score))
print("RF accuracy = {0:2.2f}, precision = {1:2.2f}, recall = {2:2.2f}, f1_score = {3:2.2f}".format(RF_metrics.accuracy, RF_metrics.precision, RF_metrics.recall, RF_metrics.f1_score))
