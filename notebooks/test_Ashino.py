#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:49:20 2021

@author: carson
"""

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
# from sktime.classification.compose import TimeSeriesForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

df1 = pd.read_csv("../data/Ashino/Data_trim.csv")
df2 = pd.read_csv("../data/Ashino/Data_trim2.csv")

df = pd.concat([df1.reset_index(drop=True), df2], axis=1).set_index('Time (h)').T

meta = pd.read_csv("../data/Ashino/meta.csv")
#sample data even time points:
# df = df.iloc[:, [0,1,2,3,7,9,11,15,17,19,23,25,27,31,33,35,37]]

print(df.head())

y = meta['Cond_num'].values


X = df
X = df.drop([1,2,3,4,5], axis = 1) #remove the first few with high error
X = np.log(X - min(X.min()) + 0.001) #optional log transformation. #adding min b/c negative values
#I add the small buffer to make there exist no 0 values after making smallest 0



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)



# let's get a baseline for comparison
classifier = DummyClassifier(strategy="prior")
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)


# now we can apply any scikit-learn RandomForest classifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)

# accuracy = 0.721. not bad! better than Knn:

knn = KNeighborsClassifier(n_neighbors=1)
# knn = svm.SVC()

knn.fit(X_train, y_train)

y_pred = knn.predict(X_train)

new_prediction = knn.predict(X_test)
print("Prediction: {}".format(new_prediction))


print(knn.score(X_test, y_test))

# knn = KNeighborsTimeSeries(n_neighbors=2)
# knn.fit()



# setup arrays to store and train and test accuracies
neighbors = np.arange(1,9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# loop over different values of k
for i, k in enumerate(neighbors):
    #setup a k-NN classifier with k nieghbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # fit the classifier to the training data
    knn.fit(X_train, y_train)

    #compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)


# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# now let's try making an RNN (doing this before finishing courses on how, should come back to it...)
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

# predictors = np.loadtxt('predictors_data.csv', delimiter=',')
# n_cols = predictors.shape[1]

# model = Sequential()
# model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1))

# predictors = X
# target = to_categorical(y)[:,1:]
# # My worked out example:

# # Save the number of columns in predictors: n_cols
# n_cols = predictors.shape[1]

# # Set up the model: model
# model = Sequential()

# # Add the first layer
# model.add(Dense(50, activation="relu", input_shape=(n_cols,)))

# # Add the second layer
# model.add(Dense(32, activation="relu", input_shape=(n_cols,)))

# # Add the output layer
# model.add(Dense(1))

# # Compile the model
# model.compile(optimizer='adam', loss = 'mean_squared_error')

# # Verify that model contains information from compiling
# print("Loss function: " + model.loss)

# # fit the mode
# model.fit(predictors, target)


## FOR CLASSIFICATION MODELS, loss = 'categorical_crossentropy' instead of mse in compile step
# also print out accuracy score with metrics = ['accuracy'] and activation = 'softmax'


# from keras.callbacks import EarlyStopping

predictors = X
target = to_categorical(y)[:,1:]
# target = y
n_cols = predictors.shape[1]

# model = Sequential()
# # Add the first layer
# model.add(Dense(15, activation="relu", input_shape=(n_cols,)))
# # Add the second layer
# model.add(Dense(10, activation="relu", input_shape=(n_cols,)))
# # Add the third layer
# model.add(Dense(10, activation="relu", input_shape=(n_cols,)))



# # Add the output layer
# model.add(Dense(225, activation = 'softmax'))
# # Compile the model
# model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# # Verify that model contains information from compiling
# print("Loss function: " + model.loss)

# early_stopping_monitor = EarlyStopping(patience=2)

# # fit the mode
# model.fit(predictors, target, validation_split = 0.33, epochs=50, callbacks = [early_stopping_monitor], use_multiprocessing=True)




# # Fit model_1
# model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# # Fit model_2
# model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# # Create the plot
# plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
# plt.xlabel('Epochs')
# plt.ylabel('Validation score')
# plt.show()






# from keras.models import load_model
# model.save('model_file.h5')
# my_model = load_model('my_model.h5')
# predictions = my_model.preict(data_to_predict_with)
# probability_true = predictions[:,1]











num_classes = 225


#let's try an RNN in keras instead of a regular NN
predictors = np.array(predictors[:])
predictors_RNN = predictors.reshape(predictors.shape[0], predictors.shape[1], 1)

from keras.layers import LSTM, GRU, Dense, Embedding, Bidirectional, Conv1D
from keras.models import Model
from keras.layers.recurrent import SimpleRNN
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten, Dense

model = Sequential()

# model.add(Embedding(10000, 225))
model.add(LSTM(225, dropout=0.2))
model.add(Dense(225, activation='softmax'))

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

model.fit(predictors_RNN, target, epochs=10, validation_split = 0.33) #, batch_size=32) <- use if data to big to fit in memory

# model.evaluate(X_test, y_test)

# model.precict(new_data)


model = Sequential()

input_shape=(predictors_RNN.shape[1], 1)

model.add(Conv1D(225, kernel_size=5, padding = 'same', activation='relu', input_shape=input_shape))

model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=(2)))
model.add(Conv1D(225,kernel_size=5,padding = 'same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=(2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.0))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit(predictors_RNN, target, epochs=100, validation_split = 0.1)
model.summary()



