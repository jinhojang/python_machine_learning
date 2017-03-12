#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 23:31:59 2017

@author: jinho
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import Perceptron
from plot_decision_regions import plot_decision_regions

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where( y == 'Iris-setosa', -1, 1 )
X = df.iloc[ 0:100, [0, 2] ].values
plt.scatter( X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa' )
plt.scatter( X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor' )
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend( loc='upper left' )
plt.show()

ppn = Perceptron( eta=0.1, n_iter=10 )
ppn.fit( X, y )
plt.plot( range( 1, len(ppn.errors_) + 1 ), ppn.errors_, marker='o' )
plt.xlabel( 'Epochs' )
plt.ylabel( 'Number of misclassifications' )
plt.show()

plot_decision_regions( X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend( loc='upper left')
plt.show()