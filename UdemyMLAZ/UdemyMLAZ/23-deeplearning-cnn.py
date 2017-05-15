import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

cwd = os.getcwd()
np.set_printoptions(threshold=np.nan)

# Importing the dataset
import sys
from os import listdir
from os.path import isfile, join

try:
    datapath = '..\\DeepLearning_CNN\\dataset\\'
    dataset = [f for f in listdir(datapath)]
except:
    datapath = 'UdemyMLAZ\\DeepLearning_CNN\\dataset\\'
    dataset = [f for f in listdir(datapath)]

training_set_path = datapath + "training_set\\"
test_set_path = datapath + "test_set\\"


# Part 1 - building the CNN

# importing the keras libs
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, Conv2D, MaxPooling2D

# initialize the CNN
classifier = Sequential()

# you improve the performance of your model by either:
#  - adding an additional convolutional layer (step 1) (you need step 1 & 2)
#  - adding an additional fully connected layer (step 4)
#  - or both

# step 1 - convolution
# create 32 feature detector 
# each feature detector/map contains 3 rows and 3 columns
# each feature detector/map will have the input_shape of 3 (for RGB) or 1 (for BW) and density of colors (e.g. 256, 128, 64, 32)
# params order for theano:
#classifier.add(Convolution2D(32, 3, 3, input_shape = (3,64,64)))
# params order for tensorflow:
#classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), activation = 'relu'))
classifier.add(Conv2D(32, (3,3), activation = "relu", input_shape = (64,64,3)))

# step 2 - max pooling
#classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(MaxPool2D(pool_size = (2,2)))

### adding second layer to improve performance
### the shape will be the one from the previous pooled feature maps
classifier.add(Conv2D(32, (3,3), activation = "relu"))
classifier.add(MaxPool2D(pool_size = (2,2)))

### you can add additional one with 64 feature maps


# step 3 - flattening
# convert all pooled feature maps to a single vector to be the input of the ANN
classifier.add(Flatten())

# step 4 - full connected layered ANN
# maybe try with 64?
classifier.add(Dense(units = 128, activation = 'relu'))

# step 5 - add output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# step 6 - compile the CNN
classifier.compile(optimizer = 'adam', 
                    loss = 'binary_crossentropy', 
                    metrics = ['accuracy'])

# Part 2 - fitting the CNN to the images

# generate more augemntations of training dataset
from keras.preprocessing.image import ImageDataGenerator

# BEGIN - code below is copied from Keras documentation
# step 1 - prepare training data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# the target size here should match your dimensions (64,64)
training_set = train_datagen.flow_from_directory(
        training_set_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# step 2 - prepare test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
        test_set_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# step 3 - fit data
# steps_per_epoch is count of training set items 8000
# nb_epoch is number of epochs you want to choose the CNN
# validation_steps is number of test set items 2000
classifier.fit_generator(
        training_set,
        steps_per_epoch = 8000,
        epochs = 25,
        validation_data = test_set,
        validation_steps = 2000)
# END - code above is copied from Keras documentation
