#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 19:25:43 2021

@author: carson
"""

# from tslearn.clustering import TimeSeriesKMeans
# from tslearn.neighbors import KNeighborsTimeSeries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

df = pd.read_csv("../data/raw/hsal/20150517_PQ_3/PQ_batch3_transposed_ctl.csv")

print(df.head())


y = df['strain'].values
X = df.drop('strain', axis = 1).values

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=1)

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





# # Now add DTW into the function to see effect
# # custom DTW metric from N Rieble on Stackoverflow..
# # https://stackoverflow.com/questions/57015499/how-to-use-dynamic-time-warping-with-knn-in-python
# def DTW(a, b):
#     an = a.size
#     bn = b.size
#     pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
#     cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
#     cumdist[0,0] = 0

#     for ai in range(an):
#         for bi in range(bn):
#             minimum_cost = np.min([cumdist[ai, bi+1],
#                                    cumdist[ai+1, bi],
#                                    cumdist[ai, bi]])
#             cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

#     return cumdist[an, bn]


# knn_dtw = KNeighborsClassifier(n_neighbors=3, metric=DTW)

# knn_dtw.fit(X_train, y_train)

# y_pred = knn_dtw.predict(X_train)

# new_prediction = knn_dtw.predict(X_test)
# print("Prediction: {}".format(new_prediction))


# print(knn_dtw.score(X_test, y_test))

# # knn = KNeighborsTimeSeries(n_neighbors=2)
# # knn.fit()



# # setup arrays to store and train and test accuracies
# neighbors = np.arange(1,9)
# train_accuracy = np.empty(len(neighbors))
# test_accuracy = np.empty(len(neighbors))

# # loop over different values of k
# for i, k in enumerate(neighbors):
#     #setup a k-NN classifier with k nieghbors: knn
#     knn_dtw = KNeighborsClassifier(n_neighbors=k, metric=DTW, n_jobs=16)

#     # fit the classifier to the training data
#     knn_dtw.fit(X_train, y_train)

#     #compute accuracy on the training set
#     train_accuracy[i] = knn_dtw.score(X_train, y_train)

#     # compute accuracy on the testing set
#     test_accuracy[i] = knn_dtw.score(X_test, y_test)


# # Generate plot
# plt.title('k-NN: Varying Number of Neighbors')
# plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
# plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
# plt.legend()
# plt.xlabel('Number of Neighbors')
# plt.ylabel('Accuracy')
# plt.show()