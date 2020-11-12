#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 13:54:26 2020

@author: jamieburns
"""
from math import floor, exp
import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    
    def __init__(self, N, min_iteration = 10, max_iteration =200):   
        """
        X: The date set
        N: The number of sources
        iteration: Number Iteration
        """
        #speccify min and maximum number of iterations to prevent over/under fitting
        self.min_iteration = min_iteration
        self.max_iteration = max_iteration
        self.N = N 
        self.cov = None
        self.mu = None
        self.lambdas = None
        #Regularization covariance
        self.reg_cov = []
        
    def e_step(self, X):
        """
        rows: The number of rows
        """
        #Initialize r (Probabilirt that pixel 'j' belongs to cluster 'i')
        r = np.zeros((len(X), self.N))
        #Calculate denominator
        for c in range(self.N):
            #update r
            r[:, c] = self.lambdas[c]*multivariate_normal(mean = self.mu[c], cov=self.cov[c]).pdf(X)
        ll = np.sum(np.log(np.sum(r, axis = 1)))
        return r / ( r.sum(axis = 1, keepdims = 1) + 1e-6), ll
    
    def m_step(self, r, X):    
        """M Step"""

        #calculate cluster weights
        c_weight = np.sum(r, axis = 0)
        #Update lambdas
        self.lambdas = c_weight / X.shape[0]
        #update mean
        weighted_sum = np.dot(r.T, X)
        self.mu = weighted_sum / c_weight.reshape(-1, 1)
        
        #update covariance
        for k in range(self.N):
            diff = (X - self.mu[k]).T
            weighted_sum = np.dot(r[:, k] * diff, diff.T)
            self.cov[k] = weighted_sum / c_weight[k]
        
        
    def run(self, X):
        _, cols  = X.shape
        #Randomly initialize mu
        #np.random.seed(100)
        self.mu = np.random.random((self.N, cols))
        #Randomly initialize cov
        self.cov = []
        for c in range(self.N):
            A = np.random.random((cols, cols))
            A *= A.T
            A += 1e+3*np.eye(cols)
            self.cov.append(A)
        self.cov = np.array(self.cov)
        #Initialzie regularization matrix
        self.reg_cov = []
        for n in range(self.N):    
            self.reg_cov.append(1e-6*np.identity(len(X[0])))
        self.reg_cov = np.array(self.reg_cov)
        #Initialize initial weigthings of each distribution
        self.lambdas = np.ones(self.N) / self.N 
        count = 0
        #Set Criteria
        ll_new = 0
        ll_old = 0
        while( self.min_iteration > count or count <= self.max_iteration ):
            #E-step
            r, ll_new = self.e_step(X)
            #M-step (update paramameters)
            self.m_step(r, X)
            #Make sure the covariance matrix is not singular
            self.cov = np.nan_to_num(self.cov) + self.reg_cov
            if (abs(ll_new - ll_old) < 1e-3 and count > self.min_iteration):
                #print('broke by convergence')
                break
            ll_old = ll_new
            
            count += 1
            
            
    def predict(self,Y):
        #get prediction
        r = np.zeros((len(Y), self.N))
        #Calculate denominator
        for c in range(self.N):
            #update r
            r[:, c] = self.lambdas[c]*multivariate_normal(mean = self.mu[c], cov=self.cov[c]).pdf(Y)
        #take dot product of prediction and lambdas
        return np.array([np.dot(p, self.lambdas) for p in r])