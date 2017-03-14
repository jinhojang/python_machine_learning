#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 23:43:44 2017

@author: jinho
"""

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

'''
C: Inverse of regularization strength - smaller values specify stronger regularization
'''
svm = SVC( kernel='linear', C=1.0, random_state=0 )
svm.fit( X_train_std, y_train )
X_combined_std = np.vstack( ( X_train_std, X_test_std ) )
y_combined = np.hstack( ( y_train, y_test ) )
plot_decision_regions( X=X_combined_std, y=y_combined,
                       target_names=iris.target_names,
                       classifier=svm,
                       test_idx=range(105, 150))
plt.title( 'Scikit-Learn LogisticRegression' )
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend( loc='lower right')
fig = plt.figure()