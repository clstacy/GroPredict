#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 17:17:54 2021

@author: carson
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import scale


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = np.array(iris.target, dtype=int)

df = pd.read_csv("~/Downloads/FASTPHAGEcleaned.csv")

#sample data even time points:
df = df.iloc[:, [0,1,2,3,7,9,11,15,17,19,23,25,27,31,33,35,37]]

print(df.head())

y = df['Key'].values
# y = df['HostConc'].values
# X = scale(np.log(df.drop(['Key', 'HostConc', 'VirusConc'], axis = 1).values)) #scale preprocessing std and mean normalized at each time point. provides very small increase to score
X = scale(df.drop(['Key', 'HostConc', 'VirusConc'], axis=1).values)
X = df.drop(['Key', 'HostConc', 'VirusConc'], axis=1)
# X = np.log(df.drop(['Key', 'HostConc', 'VirusConc'], axis = 1).values)
# note log transformation of OD values

h = 0.02 # step size in the mesh

# kernel = 1.0 * RBF([1.0])
# gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)
# kernel = 1.0 * RBF([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)

X_long = pd.melt(X).values
y_long = np.tile(y, len(X.columns))

kernel = 1.0 * RBF([1.0])
gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(X_long, y_long)
gpc_rbf_isotropic.score(X_long, y_long)


kernel = 1.0 * RBF([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel).fit(X_long, y_long)
gpc_rbf_anisotropic.score(X_long, y_long)


# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

titles = ["Isotropic RBF", "Anisotropic RBF"]
plt.figure(figsize=(10, 5))
for i, clf in enumerate((gpc_rbf_isotropic, gpc_rbf_anisotropic)):
    # Plot the predicted probabilities. For that, we will assign a color to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.subplot(1, 2, i + 1)

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
    plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y],
                edgecolors=(0, 0, 0))
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("%s, LML: %.3f" %
              (titles[i], clf.log_marginal_likelihood(clf.kernel_.theta)))

plt.tight_layout()
plt.show()