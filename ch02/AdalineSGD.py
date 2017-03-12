# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy.random import seed

class AdalineSGD( object ):
    ''' Adaptive linear neuron classifier
    
    Parameters
    eta     : learning rate
    n_iter  : number of iteration - passes over the training dataset
    
    Attributes
    w_      : weights
    errors_ : number of misclassification in every epoch
    shuffle : if Ture, suffle the training date set to prevent cycles
    random_state : random state for shuffling
    '''
    
    def __init__( self, eta=0.01, n_iter=10, shuffle=True, random_state=None ):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
             seed( random_state )
             
             
    def _initialize_weights( self, m ):
        self.w_ = np.zeros( 1 + m )
        self.w_initialized = True
             
        
    def _shuffle( self, X, y ):
        r = np.random.permutation( len( y ) )
        return X[r], y[r]
    
    
    def _update_weights( self, xi, target ):
        output = self.net_input( xi )
        error = ( target - output )
        self.w_[1:] += self.eta * xi.dot( error )
        self.w_[0:] += self.eta * error
        cost = error**2 / 2.0
        return cost
    
    
    def partial_fit( self, X, y ):
        
        if not self.w_initialized:
            self._initialize_weights( X.shape[1] )
        if y.ravel().shape[0] > 1:
            for xi, target in zip( X, y ):
                self._update_weights( xi, target )
        else:
            self._update_weights( X, y )
        
    
    def fit( self, X, y ):
        ''' fit training data set to model
        
        Parameters
        X : Training vectors, shape = [ n_samples, n_features ]
        y : Target value, shape = [ n_samples ]
        
        Returns
        self
        '''
        self._initialize_weights( X.shape[1] )
        self.cost_ = []
        
        for _ in range( self.n_iter ):
            if self.shuffle:
                X, y = self._shuffle( X, y )
            cost = []
            for xi, target in zip(X, y):
                cost.append( self._update_weights( xi, target ) )
            avg_cost = sum( cost ) / len( y )
            self.cost_.append( avg_cost )
        return self
    
    def net_input( self, X ):
        ''' compute the net input '''
        return np.dot( X, self.w_[1:] ) + self.w_[0]
    
    def activiation( self, X ):
        return self.net_input( X )
    
    def predict( self, X ):
        ''' return the predicted class label '''
        return np.where( self.activiation( X ) >= 0.0, 1, -1 )