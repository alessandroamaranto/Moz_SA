#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:30:49 2019

@author: alessandroamaranto
"""

import numpy as np
import mult_ann as ann
import policy

def getNormOutput(param, X):

    K = 4
    
    xmin = np.array([200, -1, -1, 10, 100, 100, 0])
    xmax = np.array([40000, 15, 10, 4500, 3000, 3000, 1])
    
    ymin = np.array([0.5, 0.5, 0.05, 0.05])
    ymax = np.array([0.5045454545454545, 0.5045454545454545, 0.05045, 0.05045])
    
    Y = np.ndarray(shape = (X.shape[0], K))
    
    for i in range(0, X.shape[0]):
        
        x = X[i, :]
        inp = normalize(x, xmin, xmax)
        y = np.array(ann.getOutput(inp, policy.p_param))
        Y[i, :] = getDenormalizedOut(y, ymin, ymax)
    return Y
        
    
    
def normalize(x, parmin, parmax):
    
    y = (x - parmin) /(parmax - parmin)
    return y

def getDenormalizedOut(y, ymin, ymax):
    
    z = np.ndarray(shape = y.shape[0])
    
    for i in range(0, y.shape[0]):
        z[i] = y[i]*(ymax[i] - ymin[i]) + ymin[i]
    return z


