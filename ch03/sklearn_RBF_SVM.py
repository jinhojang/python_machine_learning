# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:25:34 2017

@author: Jinho
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from plot_decision_regions import plot_decision_regions

np.random.seed(0)
X_xor = np.random.randn( 200, 2 )
y_xor = np.logical_xor( X_xor[:, 0] > 0, X_xor[:, 1]> 0 )
y_xor = np.where( y_xor, 1, -1 )
svm = SVC( kernel='rbf', random_state=0, gamma=0.10, C=10.0 )
svm.fit( X_xor, y_xor )
plot_decision_regions( X_xor, y_xor, target_names=['-1', '1', '-1'], classifier=svm )
plt.legend(loc='upper left')
plt.show()