# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

class AdalineGD( object ):
    ''' Adaptive linear neuron classifier
    
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
        self.cost_ = []
        
        for _ in range( self.n_iter ):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot( errors )
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append( cost )
        return self
    
    def net_input( self, X ):
        ''' compute the net input '''
        return np.dot( X, self.w_[1:] ) + self.w_[0]
    
    def activiation( self, X ):
        return self.net_input( X )
    
    def predict( self, X ):
        ''' return the predicted class label '''
        return np.where( self.activiation( X ) >= 0.0, 1, -1 )