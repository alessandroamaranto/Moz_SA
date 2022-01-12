#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 09:42:44 2019

@author: alessandroamaranto
"""

import numpy as np
import pset
import sa_graphics as sag
import matplotlib.pyplot as plt
import sa_utils
import pickle

def pawnSA(policy, objectives, tresholds):
    
    Y_names = ['Irrigation downstream', 'Irrigation upstream', 'Urban Deficit', 'Hydropower']
    
    '''
    Uncertainty characterization    
    '''
    
    # Create a sequence of potential streamflow data
    eQ = -0.2
    Nr = 5000
    X = sa_utils.create_perturbed_list(pset.q_in, eQ, Nr)
    mf = X[0]
    Xq = X[1]
    
    # Create a sequence of potential agricultural demand data
    eD = 0.25
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
    
    Ju = sa_utils.policy_simulate(XU, Xq, Xd, Xdd, Xi, Xcy, policy)
    Jtu = Ju[1]
    Ju = Ju[0]
    
    '''
    GLUE uncertainty analysis
    '''
    
    nO = len(objectives)
    
    u_idx = np.ndarray(shape = nO, dtype = 'object')
    #lb = np.ndarray(shape = nO, dtype = 'object')
    #ub = np.ndarray(shape = nO, dtype = 'object')
    
    Xp = np.vstack((mf[XU[:, 0].astype(int)],
                       md[XU[:, 1].astype(int)],
                       mi[XU[:, 2].astype(int)],
                       mcy[XU[:, 3].astype(int)])).T
    
    #idx2 = np.where(Xp[:, 0] < 0.92)
    #tr_min = np.array([55, 150, 15, -6500])
    #tr_max = np.array([67, 200, 20, -5750])
    
    #Uo = sa_utils.unc_onj_fun(Ju, tr_min, tr_max)
    #tresholds = (tr_max - tr_min)/2
    tresholds  = np.percentile(Ju, 5, axis = 0)
    #tresholds[0] = 5
    #tresholds[3] = 10350
    
    for v in range(0, nO):
        u_out = sa_utils.GLUE(Ju[:, v], tresholds[v], Jtu[:, :, v], False, 'leq') 
        #u_out = np.where(u_out == 1)
        #u_out = np.intersect1d(idx2, u_out)
        u_idx[v] = u_out
        #lb[v] = u_out[1]
        #ub[v] = u_out[2]
        
        fig, axis = plt.subplots(1, M, sharey = True)
        fig.suptitle(Y_names[v])
        f2, ax2 = plt.subplots(1)
        
        for i in range(0, M):
            sag.plot_scatters(axis[i], Xp[:, i], Ju[:, v], u_idx[v], X_lab[i])  
        
        
        #fig.show()

        
        titolo =  Y_names[v] + 'scatter' + '.pdf'
        titolo_p = Y_names[v] + 'parallel' + '.pdf'
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
    Jtc = np.ndarray(shape = (M, n), dtype = 'object')
    Jc = np.ndarray(shape = (M, n), dtype = 'object')
    
    for i in range(0, M):
        for k in range(0, n):
            X = sa_utils.policy_simulate(Xc[i, k], Xq, Xd, Xdd, Xi, Xcy, policy)
            Jtc[i, k] = X[1]
            Jc[i, k] = X[0]
    
    # CDF support for obj function
    YY = sa_utils.cdf_support(Jtu, Jtc, 100)
    YF = YY[0]
    
    # Time varying sensitivity index
    SI_tv = sa_utils.pawn_ks_timevarying(Jtu, JtYF)
    SI_t = SI_tv[0]
    
    for v in range(0, nO):
        sag.tvs_img(SI_t[v], X_lab, Y_names[v])
    
    # Non time-varying SA
    SI = np.ndarray(shape = nO, dtype = 'object')
    #uB = np.ndarray(shape = nO, dtype = 'object')
    #lB = np.ndarray(shape = nO, dtype = 'object')
    
    for v in range(0, nO):
        PI = sa_utils.pawn_indices(Ju, Jc, 100, v)
        SI[v] = PI[0]
        uB = PI[1]
        lB = PI[2]
        
        fig, ax = plt.subplots(1)
        sag.plot_bxp(SI[v], uB, lB, ax, M, X_lab, Y_names[v]) 
        
        titolo = Y_names[v] + 'SA_box_marrone.pdf'
        
        fig.savefig(titolo)
        
        # Saving the objects:
    with open('SA_marrone.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([mf, Xq, md, Xd, Xdd, mi, Xi, mcy, Xcy, XU, Ju, Jtu, Xp, Xc, Jtc, Jc, SI], f)
    
    
    
    
    return mf, md, mcy, XU, u_idx, xc, Xc, SI_t, SI
    
    
# Getting back the objects:
#with open('SA_downstream_policy.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     mf, Xq, md, Xd, Xdd, mi, Xi, mcy, Xcy, XU, Ju, Jtu, Xp, Xc, Jtc, Jc, SI_tv, SI = pickle.load(f)        
 

# obj0, obj1, obj2 are created here...




    
        
    
