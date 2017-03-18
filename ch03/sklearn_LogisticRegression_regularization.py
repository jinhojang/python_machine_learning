#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 23:43:44 2017

@author: jinho
"""

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
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

weights, params = [], []
for c in np.arange(-5, 5 ):
    lr = LogisticRegression( C=10**c, random_state=0 )
    lr.fit( X_train_std, y_train )
    weights.append( lr.coef_[1] )
    params.append( 10**c )
weights = np.array( weights )
plt.plot( params, weights[:, 0], label='peta length')
plt.plot( params, weights[:, 1], label='peta width', linestyle='--')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.show()