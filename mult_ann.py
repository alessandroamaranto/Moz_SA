#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:06:51 2019

@author: alessandroamaranto
"""

import numpy as np
import policy


class Obj(object):
    pass

def setParameters(pTheta):
    
    param = Obj()
    param.d = np.ndarray(shape = (policy.K, policy.N), dtype = 'object')
    param.a = np.ndarray(shape = policy.K)
    param.b = np.ndarray(shape = (policy.K, policy.N), dtype = 'object')
    param.c = np.ndarray(shape = (policy.K, policy.M, policy.N), dtype = 'object')

    idx_theta = 0;
    
    for k in range(0, policy.K):
        
        for n in range(0, policy.N):
            param.d[k, n] = pTheta[idx_theta]
            
            idx_theta = idx_theta + 1
            
        param.a[k] = pTheta[idx_theta]
        idx_theta = idx_theta + 1
        
        for n in range(0, policy.N):
            param.b[k, n] = pTheta[idx_theta]
            idx_theta = idx_theta + 1
        
        for m in range (0, policy.M):
            for n in range(0, policy.N):
                param.c[k, m, n] = pTheta[idx_theta]
                idx_theta = idx_theta + 1
    
    return param

def getOutput(inp, pTheta):
    
    param = setParameters(pTheta)
    neurons = []

    y = []
    
    
    for k in range(0, policy.K):
        for n in range(0, policy.N):
            value  = param.d[k, n]
            for m in range(0, policy.M):
                value = value + param.c[k, m, n]*inp[m]
            value = 2 / ( 1 + np.exp(-2 * value) ) - 1
            neurons.append(value)
        
        o = param.a[k]
        
        for n in range(0, policy.N):
            o = o + param.b[k, n]*neurons[n]
        
        y.append(o)  
        neurons = []

    return y;
    
    
        
    


