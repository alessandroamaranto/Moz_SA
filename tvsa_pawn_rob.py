#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:27:36 2020

@author: alessandroamaranto
"""

import numpy as np
import sa_graphics as sag
import matplotlib.pyplot as plt
import sa_utils
import pickle

def pawnSA(nvars, nobjs, rob_out_str, filename, pn):
    
    '''
    Load optimization and robustness outputs
    '''
    
    with open(rob_out_str, 'rb') as f:  # Python 3: open(..., 'wb')
         mf, Xq, md, Xd, Xdd, mi, Xi, mcy, Xcy, XU, Ju, RI, BW, BB = pickle.load(f)
    
    
    theta = np.loadtxt(filename)
    policies = theta[:, range(nvars)]
    objectives = theta[:, nvars:]
    
    policy = policies[pn, :]
    Ju = np.loadtxt('policy_3' + str(pn) + '.txt')
    
    
    '''
    General settings
    '''
    
    Y_names = ['Irrigation downstream', 'Irrigation upstream', 'Urban Deficit', 'Hydropower']
    X_lab = ['Inflow', 'Agriculture', 'Infrastructure', 'Population']
    
    M = len(X_lab)
    Nr = 5000
    Ns = np.tile(Nr - 1, len(X_lab))
    
    '''
    GLUE uncertainty analysis
    '''
    
    nO = objectives.shape[1]
    
    u_idx = np.ndarray(shape = nO, dtype = 'object')
    #lb = np.ndarray(shape = nO, dtype = 'object')
    #ub = np.ndarray(shape = nO, dtype = 'object')
    
    Xp = np.vstack((mf[XU[:, 0].astype(int)],
                       md[XU[:, 1].astype(int)],
                       mi[XU[:, 2].astype(int)],
                       mcy[XU[:, 3].astype(int)])).T
    
    tresholds  = np.percentile(Ju, 5, axis = 0)
    
    for v in range(0, nO):
        u_out = sa_utils.GLUE(Ju[:, v], tresholds[v], False, False, 'leq') 
        #u_out = np.where(u_out == 1)
        #u_out = np.intersect1d(idx2, u_out)
        u_idx[v] = u_out
        #lb[v] = u_out[1]
        #ub[v] = u_out[2]
        
        fig, axis = plt.subplots(1, M, sharey = True)
        fig.suptitle(Y_names[v])
        f2, ax2 = plt.subplots(1)
        
        for i in range(0, M):
            sag.plot_scatters(axis[i], Xp[:, i], Ju[:, v], u_idx[v], X_lab[i], BW, pn, i, v)  
        
        
        #fig.show()

        
        titolo =  Y_names[v] + 'scatter' + str(pn) + '.pdf'
        titolo_p = Y_names[v] + 'parallel' + str(pn)  + '.pdf'
        fig.savefig(titolo)
        sag.parallel_nor(f2, ax2, Xp, u_idx[v], X_lab, Y_names[v])
        #sag.parallel(Xp, u_idx[v], X_lab, Y_names[v], 'nonnumber')
        f2.savefig(titolo_p)
        f2.show()
    
    '''
    PAWN sensitivity analysis
    '''
    
    # Conditional sampling 
    n = 12
    NC = 1000
    Xc = sa_utils.pawn_sampling('rsu', M, 'unid', Ns, n, NC)
    xc = Xc[1]
    Xc = Xc[0]
    
    # Conditional policy evaluation
    Jc = np.ndarray(shape = (M, n), dtype = 'object')
    
    for i in range(0, M):
        for k in range(0, n):
            X = sa_utils.policy_simulate(Xc[i, k], Xq, Xd, Xdd, Xi, Xcy, policy)
            Jc[i, k] = X[0]
    
    # Non time-varying SA
    SI = np.ndarray(shape = nO, dtype = 'object')
    
    for v in range(0, nO):
        PI = sa_utils.pawn_indices(Ju, Jc, 100, v)
        SI[v] = PI[0]
        uB = PI[1]
        lB = PI[2]
        
        fig, ax = plt.subplots(1)
        sag.plot_bxp(SI[v], uB, lB, ax, M, X_lab, Y_names[v], BW, pn) 
        
        titolo = Y_names[v] + 'SA_box' + str(pn) + '.pdf'
        
        fig.savefig(titolo)
        
    sao = 'SA' + str(pn) + '.pkl'
    # Saving the objects:
    with open(sao, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([mf, Xq, md, Xd, Xdd, mi, Xi, mcy, Xcy, XU, Ju, Xp, Xc, Jc, SI], f)
    
    
    
    
    return mf, md, mcy, XU, u_idx, xc, Xc, SI