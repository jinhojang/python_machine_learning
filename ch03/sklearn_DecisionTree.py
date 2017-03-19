#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 23:43:44 2017

@author: jinho
"""

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

'''
Instantiate a tree
'''
tree = DecisionTreeClassifier( criterion='entropy', max_depth=3, random_state=0,)
tree.fit( X_train, y_train )
X_combined = np.vstack( ( X_train, X_test ) )
y_combined = np.hstack( ( y_train, y_test ) )
plot_decision_regions( X=X_combined, y=y_combined,
                       target_names=iris.target_names,
                       classifier=tree,
                       test_idx=range(105, 150))
plt.title( 'Scikit-Learn DecisionTree' )
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend( loc='upper left')
fig = plt.figure()

from sklearn.tree import export_graphviz
export_graphviz( tree, out_file='tree.dot', feature_names=['setal length', 'petal length'] )

#> dot -Tpng tree.dot -o tree.png
#> feh tree.png