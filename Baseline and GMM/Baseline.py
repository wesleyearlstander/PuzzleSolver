#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 14:56:17 2020

@author: jamieburns
"""

import numpy as np
from scipy.stats import multivariate_normal


class Baseline:
    def __init__(self):
        self.mean = None
        self.Cov = None
        
    def fit(self, X):
        self.mean  = np.mean(X, axis = 0)
        self.cov = np.cov(X.T)
        
    def predict(self, Y):
        return multivariate_normal(mean = self.mean, cov = self.cov).pdf(Y)
    
        