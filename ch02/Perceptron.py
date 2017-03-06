# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

class Perceptron( object ):
    ''' Perceptron classifier
    
    Parameters
    eta     : learning rate
    n_iter  : number of iteration - passes over the training dataset
    
    Attributes
    w_      : weights
    errors_ : number of misclassification in every epoch
    '''
    
    def __init__( self, eta=0.01, n_iter=10 ):
        self.eta = eta
        self.n_iter = n_iter
        
    def fit( self, X, y ):
        ''' fit training data set to model
        
        Parameters
        X : Training vectors, shape = [ n_samples, n_features ]
        y : Target value, shape = [ n_samples ]
        
        Returns
        self
        '''
        self.w_ = np.zeros( 1 + X.shape[1] )
        self.errors_ = []
        
        for _ in range( self.n_iter ):
            errors = 0
            for xi, target in zip( X, y ):
                update = self.eta * ( target - self.predict( xi ) )
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int( update != 0.0 )
            self.errors_.append( errors )
        return self
    
    def net_input( self, X ):
        ''' compute the net input '''
        return np.dot( X, self.w_[1:] ) + self.w_[0]
    
    def predict( self, X ):
        ''' return the predicted class label '''
        return np.where( self.net_input( X ) >= 0.0, 1, -1 )