#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 23:43:44 2017

@author: jinho
"""

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

'''load iris data'''
iris =  datasets.load_iris()
X = iris.data[ :, [2, 3] ]
y = iris.target

''' split data into train and test data '''
X_train, X_test, y_train, y_test = train_test_split( X, y, 
                                                     test_size=0.3, 
                                                     random_state=0 )
''' standarized the data '''
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

''' Perceptron instance '''
ppn = Perceptron( n_iter=40, eta0=0.01, random_state=0 )
ppn.fit( X_train_std, y_train )
y_pred = ppn.predict( X_test_std )
print('Miscalssified samples: %d' % (y_test != y_pred).sum() )
print('Accuracy: %f' % accuracy_score(y_test, y_pred) )

X_combined_std = np.vstack( ( X_train_std, X_test_std ) )
y_combined = np.hstack( ( y_train, y_test ) )
plot_decision_regions( X=X_combined_std, y=y_combined,
                       target_names=iris.target_names,
                       classifier=ppn,
                       test_idx=range(105, 150))
plt.title( 'Scikit-Learn Perceptron' )
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend( loc='upper left')
fig = plt.figure()