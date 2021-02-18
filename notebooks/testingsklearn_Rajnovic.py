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
from sklearn import svm
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
from sklearn.preprocessing import scale

df = pd.read_csv("~/Downloads/FASTPHAGEcleaned.csv")


print(df.head())


# y = df['Key'].values
y = df['HostConc'].values
X = scale(np.log(df.drop(['Key', 'HostConc', 'VirusConc'], axis = 1).values)) #scale preprocessing std and mean normalized at each time point. provides very small increase to score
# X = np.log(df.drop(['Key', 'HostConc', 'VirusConc'], axis = 1).values)
# note log transformation of OD values

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42, stratify=y)

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





# df = pd.read_csv("~/Downloads/FASTPHAGEcleaned_SelectedComparisons.csv")

# print(df.head())


# y = df['Key'].values
# # y = df['HostConc'].values
# X = np.log(df.drop(['Key', 'HostConc', 'VirusConc'], axis = 1).values)
# # note log transformation of OD values

# # Split into training and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42, stratify=y)

# knn = KNeighborsClassifier(n_neighbors=1)

# knn.fit(X_train, y_train)

# y_pred = knn.predict(X_train)

# new_prediction = knn.predict(X_test)
# print("Prediction: {}".format(new_prediction))


# print(knn.score(X_test, y_test))

# # knn = KNeighborsTimeSeries(n_neighbors=2)
# # knn.fit()



# # setup arrays to store and train and test accuracies
# neighbors = np.arange(1,9)
# train_accuracy = np.empty(len(neighbors))
# test_accuracy = np.empty(len(neighbors))

# # loop over different values of k
# for i, k in enumerate(neighbors):
#     #setup a k-NN classifier with k nieghbors: knn
#     knn = KNeighborsClassifier(n_neighbors=k)

#     # fit the classifier to the training data
#     knn.fit(X_train, y_train)

#     #compute accuracy on the training set
#     train_accuracy[i] = knn.score(X_train, y_train)

#     # compute accuracy on the testing set
#     test_accuracy[i] = knn.score(X_test, y_test)


# # Generate plot
# plt.title('k-NN: Varying Number of Neighbors')
# plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
# plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
# plt.legend()
# plt.xlabel('Number of Neighbors')
# plt.ylabel('Accuracy')
# plt.show()






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
#                                     cumdist[ai+1, bi],
#                                     cumdist[ai, bi]])
#             cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

#     return cumdist[an, bn]


# knn_dtw = KNeighborsClassifier(n_neighbors=1, metric=DTW, n_jobs=-1)

# knn_dtw.fit(X_train, y_train)

# y_pred = knn_dtw.predict(X_train)

# new_prediction = knn_dtw.predict(X_test)
# print("Prediction: {}".format(new_prediction))


# print(knn_dtw.score(X_test, y_test))



# # setup arrays to store and train and test accuracies
# neighbors = np.arange(1,4)
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











#try fitting regression model to these data?


def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha

    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)

    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))

    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)













# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import classification_report#, train_test_split

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))