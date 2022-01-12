#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:59:10 2019

@author: alessandroamaranto
"""

import numpy as np
import pandas as pd
import rbf
import platform

def Op_Sys_Folder_Operator():
    '''
    Function to determine whether operating system is (1) Windows, or (2) Linux
 
    Returns folder operator for use in specifying directories (file locations) for reading/writing data pre- and
    post-simulation.
    '''
 
    if platform.system() == 'Windows':
        os_fold_op = '\\'
    elif platform.system() == 'Linux':
        os_fold_op = '/'
    else:
        os_fold_op = '/'  # Assume unix OS if it can't be identified
 
    return os_fold_op

def bounds(M, N, K, irr):  
    a = np.concatenate((np.tile([0, 1], K),
                        np.tile(
                                np.concatenate((
                                        np.tile([-1, 1, 0, 1], M),
                                        np.tile([0, 1], K)))
                                , N)))
    bnd = []
    l = 0
    for i in range(int(len(a)/2)):
        b = a[l:(l+2)]
        b = b.tolist()
        bnd.append(b)
        l = l + 2
    
    if irr == 'yes':
        bnd.append([np.finfo(float).eps, 1])
        bnd.append([0, 2])
        
    return bnd

def matrix2vec(filepath, sp):

    x = pd.read_csv(filepath, sep = sp)
    x = np.array(x.iloc[:, 1:])
    x = np.reshape(x, (x.shape[0]*x.shape[1], 1))
    
    return x

def monthly2daily(x, start, end):
    
    c = 3600*24
    dt = np.arange(start, end, dtype='datetime64[D]')

    dates = pd.DatetimeIndex(dt)

    mu = np.unique(dates.month)
    yu = np.unique(dates.year)
    m = dates.month
    y = dates.year

    xd = np.zeros(len(dates))
    k = 0

    for j in yu:
        for i in mu:
        
            idx1 = np.where(y == j) 
            idx2 = np.where(i == m)
            idx = np.intersect1d(idx1, idx2)
        
            xd[idx] = x[k]/len(idx)/c
            k = k + 1
            
    return xd

def get_norm_output(inp, rTheta, par, r_max, irr_max):
  
  y= []   
  inp = normalize(inp, par)
  out = rbf.getOutput(inp, rTheta)
  
  y.append(out[0]*r_max)
  y.append(out[1]*irr_max)
  return y

def normalize(x, par):
    
    y = (x - par[0]) /(par[1] - par[0])
    return y

def getOptRes(res, nvars, nobjs):
    
    count = 0
    for re in res:
        count = count + 1
    
    sol = np.empty([count, nvars])
    ob = np.empty([count, nobjs])
    
    i = 0
    for solution in res:
        ob[i, :] = solution.getObjectives()
	    
        sol[i, :] = solution.getVariables()
        i = i + 1
    
    sol_d = pd.DataFrame(sol)
    obj_d = pd.DataFrame(ob)
    
    theta = [sol_d, obj_d]
    return theta

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:20:38 2019

@author: alessandroamaranto
"""

def power_equity(filename, treshold):
    
    theta_o = np.loadtxt(filename)
    obj = theta_o[:, range(64, 68)]
    
    idx = np.where(obj[:, 1] < treshold)[0]
    obj = obj[idx, :]*(-1)
    #obj = obj*(-1)
    
    
    obj = Normalize_array(obj)
    w = np.array([0.2, 0.2, 0.2, 0.4])
    
    nS = obj.shape[0]
    nP = obj.shape[1]
    alpha = np.ndarray(shape = obj.shape)
    
    for s in range(0, nS):
        
        xs = obj[s, :] 
        den = 0
        
        for i in range(0, nP):
            den = den + w[i]*(1 - xs[i])
        
        for i in range(0, nP):
            num = w[i]*(1 - xs[i])
            alpha[s, i] = num/den
        
    theta = np.std(alpha, axis = 1)
    alpha_m = np.mean(alpha, axis = 1)    
    
    CV = theta/alpha_m
    
    theta_c = np.where(CV == min(CV))[0]

    
    return theta_c  
    
def Normalize_array(x):
    
    xmin = np.min(x, axis = 0)
    xmax = np.max(x, axis = 0)
    
    z = (x - xmin)/(xmax - xmin)
    
    return z
    
    

