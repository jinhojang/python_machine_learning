#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 13:08:26 2017

@author: jinho
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from AdalineGD import AdalineGD
from plot_decision_regions import plot_decision_regions

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where( y == 'Iris-setosa', -1, 1 )
X = df.iloc[ 0:100, [0, 2] ].values
fig, ax = plt.subplots( nrows = 1, ncols = 2, figsize=(8, 4) )

ada1 = AdalineGD( n_iter = 10, eta = 0.01).fit( X, y )
ax[0].plot( range(1, len(ada1.cost_) + 1) , np.log10( ada1.cost_), marker='o')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title( 'Adaline - Learning rate 0.01' )

ada2 = AdalineGD( n_iter = 10, eta = 0.0001).fit( X, y )
ax[1].plot( range(1, len(ada2.cost_) + 1) , np.log10( ada2.cost_), marker='o')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title( 'Adaline - Learning rate 0.0001' )
plt.show()

#Standaraization
X_std = np.copy(X)
X_std[:,0] = ( X_std[:,0] - X_std[:,0].mean() ) / X_std[:,0].std()
X_std[:,1] = ( X_std[:,1] - X_std[:,1].mean() ) / X_std[:,1].std()

ada = AdalineGD( n_iter = 15, eta=0.01 )
ada.fit( X_std, y )
plot_decision_regions( X_std, y, classifier=ada )
plt.title( 'Adaline - Gradient Descent' )
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend( loc='upper left')
plt.show()

plt.plot( range(1, len(ada.cost_) + 1) , np.log10( ada.cost_), marker='o')
plt.xlabel('Epochs')
plt.ylabel('log(Sum-squared-error)')
plt.show()