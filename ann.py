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
    param.d = np.ndarray(shape = policy.N)
    param.c = np.ndarray(shape = (policy.M, policy.N))
    param.a = np.ndarray(shape = policy.K)
    param.b = np.ndarray(shape = (policy.N, policy.K))
    
    idx_theta = 0;
  
        
    for n in range(0, policy.N):
        param.d[n] = pTheta[idx_theta];
        idx_theta = idx_theta + 1;
    

    for n in range (0, policy.N):
        for m in range(0, policy.M):
            param.c[m, n] = pTheta[idx_theta];
            idx_theta = idx_theta + 1;
        
    

    for k in range(0, policy.K):
        param.a[k] = pTheta[idx_theta];
        idx_theta = idx_theta + 1
    
    
    for k in range(0, policy.K):
        for n in range(0, policy.N):
            param.b[n, k] = pTheta[idx_theta];
            idx_theta = idx_theta + 1;
    
    return param

def getOutput(inp, pTheta):
    
    param = setParameters(pTheta)
    neurons = []
    o = 0
    y = []
    
    for n in range(0, policy.N):
        value = param.d[n];
        for m in range(0, policy.M):
            value = value + (param.c[m, n] * inp[m]);
              
        value = 2/( 1 + np.exp(-2 * value)) - 1;
        neurons.append(value);
    
    for k in range(0, policy.K):
        o = param.a[k];
        for n in range(0, policy.N):
            o = o + param.b[n, k] * neurons[n] ;
        
        y.append(o)
    neurons = []
    

    return y;
    
    
        
    


