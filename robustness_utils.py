#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:28:21 2020

@author: alessandroamaranto
"""

import numpy as np
import sa_utils
import pset
import pickle

def robustness(filename, nvars, nobjs):
    
    theta = np.loadtxt('out/sets/' + filename)
    policies = theta[:, range(nvars)]
    objectives = theta[:, nvars:]
    
    pS = policies.shape[0]
    
    '''
    Uncertainty characterization    
    '''
    
    # Create a sequence of potential streamflow data
    eQ = -0.4
    Nr = 5000
    X = sa_utils.create_perturbed_list(pset.q_in, eQ, Nr)
    mf = X[0]
    Xq = X[1]
    
    # Create a sequence of potential agricultural demand data
    eD = 0.35
    X = sa_utils.create_perturbed_list(pset.du, eD, Nr)
    md = X[0]
    Xd = X[1]
    
    Nsteps = len(pset.dd)
    Xdd = np.ndarray(shape  = (Nsteps, Nr))
    for i in range(0, Nr):
        Xdd[:, i] =  pset.dd*md[i] 
        
    # Create a sequence of potential external inflow data
    eI = 1.8
    Nsteps = Xd.shape[0]
    Xi = sa_utils.create_constant_list(Nsteps, eI, Nr)
    mi = Xi[1]
    Xi = Xi[0]
    
    # Create a list of urban water demands data
    eCy = 0.02
    X = sa_utils.create_growing_list(pset.Cd, eCy, Nr, Nsteps)
    mcy = X[0]
    Xcy = X[1]
    
    '''
    Unconditional model evaluation
    '''
    
    # Additional parameters
    X_lab = ['Inflow', 'Agricolture', 'Infrastructure', 'Population']
    M = len(X_lab)
    Ns = np.tile(Nr - 1, len(X_lab))
    NU = 2000
    
    # Unconditional sampling and unconditional policy evaluation
    XU = sa_utils.AAT_sampling('rsu', M, 'unid', Ns, NU)
    RI = np.ndarray(shape = (pS, 6))
    BW = np.ndarray(shape = (pS, 4))
    BB = np.ndarray(shape = (pS, 4))
    
    for p in range(0, pS):
        print(p)
        policy = policies[p, :]
        
        Ju = sa_utils.policy_simulate(XU, Xq, Xd, Xdd, Xi, Xcy, policy)
        Ju = Ju[0]
        
        name = 'policy_3' + str(p) + '.txt'
        np.savetxt(name, Ju)
        
        RI[p, :] = satisfaction(Ju, objectives)
        BW[p, :] = min_max(Ju)
        BB[p, :] = min_min(Ju)
        
    with open('Robustness.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([mf, Xq, md, Xd, Xdd, mi, Xi, mcy, Xcy, XU, Ju, RI, BW, BB], f)
    
        
    return RI, BW, BB

def regret_expected(uJ, bJ):
    
    dS = np.ndarray(shape = uJ.shape)
    
    for i in range(0, uJ.shape[0]):

        dS[i, :] = abs((uJ[i, :] - bJ))/uJ[i, :]
    rS  = np.percentile(dS, 90, axis = 0)
    r = max(rS)
    
    return r

def satisfaction(uJ, bJ):
    
    tr  = np.percentile(bJ, 25, axis = 0)
    bS = tr - uJ
    
    s_mv  = np.sum(bS >= 0)/uJ.shape[0]
    s_mv3 = np.sum(bS[:, 1:] >= 0)/uJ.shape[0]
    s_sv = np.sum(bS >= 0, axis = 0)/uJ.shape[0]
    
    s = np.insert(s_sv, 0 , s_mv)
    s = np.insert(s, 0, s_mv3)

    return s


def min_max(uJ):
    
    I = np.max(uJ, axis = 0)
    return I

def min_min(uJ):
    
    I = np.min(uJ, axis = 0)
    
    return I

def non_robust_idx(R):
    
    dR = np.ndarray(shape = (94, R.shape[1]) )
    
    for j in range(0, dR.shape[0]):
        rsi = j
        for i in range(0, dR.shape[1]):
             
            sol = R[rsi, i]
            x = np.sort(R[:, i])
             
            dR[j, i] = np.where(x == sol)[0].astype(int)[0]
            
    
    sm = np.sum(dR, axis = 1)
    
    sm_idx = np.where(sm == max(sm))
    sm[sm_idx] = 0
    
    sm_idx = np.where(sm == max(sm))
    sm[sm_idx] = 0
    
    sm_idx = np.where(sm == max(sm))
    return(sm_idx)
    
    
def select_robust_solutions(R, idx):
     
    rs = np.argmin(R, axis = 0)
    rs = np.hstack((rs, idx))
    dR = np.ndarray(shape = (rs.shape[0], R.shape[1]) )
     
    for j in range(0, rs.shape[0]):
        rsi = rs[j]
        for i in range(0, dR.shape[1]):
             
            sol = R[rsi, i]
            x = np.sort(R[:, i])
             
            dR[j, i] = np.where(x == sol)[0].astype(int)[0]
            
    cols = [1, 3, 2, 0]
        
    dR = dR[:, cols]
    
    return dR

        
      
      
     