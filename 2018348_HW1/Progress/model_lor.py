#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


class LogisticRegression:
    
    def __init__(self,n_iter = 1000,l_rate = 0.1):
        self.n_iter = n_iter   # Number of iterations for gradient descent
        self.l_rate = l_rate  # Learning rate to be used
        self.wt     = None    # Weights for a particular class instance / Regressor
        self.b      = None    # Bias for a particular class instance / Regressor
    
    def sigmoid(self, Z):
        '''
        Implementation of Sigmoid function , which will be used while predicting y for a particular
        data point.
        '''
        return (1/ (1+ np.exp(-Z) ) )
        
    def fit(self,X,Y):

        n_samples  = X.shape[0] # Variable to store the number of samples in the given training sample
        n_features = X.shape[1] # Variable to store the number of features in the given training sample
        
        print(n_samples,n_features)
        
        self.wt = np.zeros(n_samples)
        self.b  = 0
        
        for iter_ in range(self.n_iter):
            
            Z = np.dot(X,self.wt) + b
            y_pred = self.sigmoid(Z)
            
            self.wt += self.l_rate *((np.dot(X.T,y-y_pred))/n_samples)
            self.b  += self.l_rate *((np.sum(y-y_pred))/n_samples)
        
    def predict(self,X):
        Z = np.dot(X,self.wt) + b
        y_pred = self.sigmoid(Z)
        return y_pred
    
        


# In[ ]:




