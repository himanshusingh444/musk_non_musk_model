#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:59:26 2020

@author: himanshu
"""

# Artificial Neural Network



# - Data Preprocessing

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

# Importing the dataset
dataframe = pd.read_csv('musk_csv.csv')
#breaking dataframe into features and y vectors

x = dataframe.iloc[:, 3:169].values
y = dataframe.iloc[:, 169].values



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 84, kernel_initializer = 'uniform', activation = 'relu', input_dim = 166)) #units perimetr is 84 because we take average of input and output nodes

# Adding the second hidden layer
classifier.add(Dense(units = 84, kernel_initializer = 'uniform', activation = 'relu'))#we have used rectifier function in hidden layer

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) #we have use sigmoid function in output layer because our problem is on the basis of classifiaction

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
                                      #if outcome is not binary then we will use 'categorical_crossentropy loss functiom'

# Fitting the ANN to the Training set

history = classifier.fit(x_train, y_train, batch_size = 20, epochs = 80, validation_data=(x_test,y_test))

# Part 3 - Making predictions and evaluating the model

y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)# more than 50% means prediction is correct
print(history.history.keys())



#visualisation part
#accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#loss graph

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix #it tells the accuracy and inaccurate result of our predicted value
cm = confusion_matrix(y_test, y_pred)

# getting accuracy score

from sklearn.metrics import accuracy_score
print('Accuracy',accuracy_score(y_test,y_pred))

#getting f1_score

from sklearn.metrics import f1_score
print('f1 score',f1_score(y_test,y_pred))

#performance of our model

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))