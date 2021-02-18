#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:49:20 2021

@author: carson
"""


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

df1 = pd.read_csv("../data/Ashino/Data_trim.csv")
df2 = pd.read_csv("../data/Ashino/Data_trim2.csv")

df = pd.concat([df1.reset_index(drop=True), df2], axis=1)

meta = pd.read_csv("../data/Ashino/meta.csv")
#sample data even time points:
# df = df.iloc[:, [0,1,2,3,7,9,11,15,17,19,23,25,27,31,33,35,37]]

print(df.head())

y = meta['Cond_num'].values

# X = scale(df.drop(['Key', 'HostConc', 'VirusConc'], axis=1).values)
X = df.drop(['Key', 'HostConc', 'VirusConc'], axis=1)
# X = np.log(df.drop(['Key', 'HostConc', 'VirusConc'], axis = 1).values)
# note log transformation of OD values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# let's get a baseline for comparison
classifier = DummyClassifier(strategy="prior")
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)


# now we can apply any scikit-learn classifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)



# binary target variable
labels, counts = np.unique(y_train, return_counts=True)
print(labels, counts)
fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))
for label in labels:
    X_train.loc[y_train == label, "dim_0"].iloc[0].plot(ax=ax, label=f"class {label}")
plt.legend()
ax.set(title="Example time series", xlabel="Time");

