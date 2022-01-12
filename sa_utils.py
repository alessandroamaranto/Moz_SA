#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:55:52 2019

@author: alessandroamaranto
"""

import numpy as np
from math import pi
import pset
import utls
import lake_bpl as bpl
import pyDOE 
from statsmodels.distributions.empirical_distribution import ECDF

def create_perturbed_list(q, eQ, Nr):
    
    Nsteps = len(q)
    Xq = np.ndarray(shape  = (Nsteps, Nr))
    mult = np.ndarray(shape  = Nr)
    
    for i in range(0, Nr):
        
        mult[i] = (1+eQ*np.random.uniform(0.25, 1))
        Xq[:, i] =  q*mult[i] #perturbe_inflow(q, eQ)
        
    return mult, Xq

def create_constant_list(Nsteps, eI, Nr):
    
    Xi = np.ndarray(shape  = (Nsteps, Nr))
    idx = np.random.randint(low = 0, high = Nsteps, size = Nr)
    
    for i in range(0, Nr):
        Xi[:idx[i], i] = 0
        Xi[idx[i]:, i] = 1.8

    return Xi, idx

def create_growing_list(Cd, eU, Nr, Nsteps):
    
    nY = round(Nsteps/365, 0)
    mult = np.random.uniform(low = 0, high = eU, size = Nr)
    mY = Cd*nY*mult + Cd
    
    Xp = np.ndarray(shape  = (Nsteps, Nr))
    Xp[0, :] = Cd
    Xp[Nsteps-1,:] = mY
    
    for i in range(0, Nr):
        d = np.linspace(Cd, Xp[Nsteps-1, i], Nsteps)
        Xp[:, i] = d
    return mult, Xp     

def AAT_sampling(samp_strat, M, distr_fun, distr_par, N):
    
    if samp_strat == 'rsu':
        X = np.random.uniform(low = 0, high = 1, size = (N, M))
    elif samp_strat == 'lhs':
        X = pyDOE.lhs(M, N)
    else:
        raise Exception('sampling strategy has to be either rsu or lhs')
        
    tmp =  np.ndarray(shape=(M), dtype = 'U25')
    for i in range(0, M):
        tmp[i] = distr_fun
    
    distr_fun = tmp
        
    for i in range(0, M):
        
        pars = distr_par[i]
        name = distr_fun[i]
        
        if  name == 'unid':
            
            X[:, i] = np.ceil(X[:, i] * pars)
        
        elif name == 'unif':
            
            X[:, i] = pars[0] + X[:, i]*(pars[1] - pars[0])
            
        else:
            raise Exception('unid is the only distribution implemented so far ')
            
    return X

def policy_simulate(X, Xq, Xd, Xdd, Xi, Xcy, policy):
    
    for i in range(0, X.shape[1]):
    
        cl = X[:, i]
        mn = np.nanmean(cl)
    
        cl[np.isnan(cl)] = mn
        
        X[:, i] = cl
    
    # Initialize SA out and par
    N = X.shape[0]
    H = Xq.shape[0]
    J = np.ndarray(shape = (N, H, 4))
    Ja = np.ndarray(shape = (N, 4))
    
    # Initialize policy out and par
    idx = pset.N*(2*pset.M + pset.K) + pset.K
    theta_r = policy[:idx]
    theta_d = policy[idx:]   
    
    for j in range(0, N):
    
        nY = H/pset.T
        
        # Initialize trajectories
        s = np.zeros(H + 1)
        h = np.zeros(H + 1)
        doy = np.zeros(H)
        
        # Decision variables
        u = np.zeros(H)
        uf = np.zeros(H)
        rfd = np.zeros(H + 1)
        rc = np.zeros(H + 1)
        rd = np.zeros(H + 1)
        ru = np.zeros(H + 1)
        
        # Initial conditions
        h[0] = pset.I[0]
        s[0] = bpl.level2storage(h[0])
        
        # Model forcings
        Q = Xq[:, int(X[j, 0]) ]
        dd = Xdd[:, int(X[j, 1])]
        du = Xd[:, int(X[j, 1])]
        Ia = Xi[:, int(X[j, 2])]
        Cd = Xcy[:, int(X[j, 3])]
        
        # Objective function and ts objective
        JJ = []   
        
        for t in range(0, H):
            
            doy[t] = (pset.I[1] + t - 1)%pset.T + 1
            
            inp = np.array([np.sin( 2*pi*doy[t]/pset.T),  
                   np.cos(2*pi*doy[t]/pset.T),  
                   h[t],
                   Q[t]
                   ])
            '''
            Release decision
            '''
            
            uu = utls.get_norm_output(inp, theta_r, pset.bn, pset.r_max, pset.irr_max)
            u[t] = uu[0]
            uf[t] = uu[1]
            
            sd_rd = bpl.integration_daily(pset.Is,
                                               t,
                                               s[t],
                                               u[t],
                                               uf[t],
                                               Q[t],
                                               doy[t],
                                               pset.MEF[t],
                                               0,
                                               0,
                                               pset.A
                                              )    
    
            s[t + 1] = sd_rd[0]
            rd[t + 1] = sd_rd[1]
            ru[t + 1] = sd_rd[2]
            
            h[t + 1] = bpl.storage2level(s[t + 1])
        
            rfd[t + 1] = min(
                    bpl.irr_output(theta_d, rd[t + 1], pset.i_bn , dd[t]),
                    max(rd[t + 1] - pset.MEF[t], 0)
                    )
            rc[t + 1] = max(rd[t] - rfd[t] - pset.MEF[t], 0)
            
            J[j, t, 0] = max(dd[t] - rfd[t + 1], 0)
            J[j, t, 1] = max(du[t] - ru[t + 1], 0)
            J[j, t, 2] = max(Cd[t] - (Ia[t] + rc[t + 1]) , 0)
            J[j, t, 3] = bpl.energy_production(rd[t + 1], h[t + 1], pset.q_turb, pset.h_down)
            
            
        rd = rd[1:]
        ru = ru[1:]
        rfd = rfd[1:]
        rc = rc[1:]
        
        '''
        Time-aggregation of the Objectives
        '''
        
        Id = IrrDeficit(rfd, dd, nY)
        Iu = IrrDeficit(ru, du, nY)
        
        JJ.append(
                Id
                )
        
        JJ.append(
                Iu
                )
        
        JJ.append(
                CityDeficit(rc, Ia, Cd, nY)
                )
        
        JJ.append(
                HydroPower(J[j, :, 3], nY)
                )
    
        Ja[j, :] = np.array(JJ)
        
    return Ja, J
        
    
def IrrDeficit(ud, Wd, nY):
    

    d = Wd - ud
    d[d < 0] = 0
    wq = np.quantile(Wd, 0.7)
    
    idx = Wd > wq
    d[idx] = d[idx]*2
    
    Jirr = sum(d)/nY
    
    return Jirr

def CityDeficit(uc, Ia, C, nY):
    
    d = C - uc - Ia
    d[d < 0] = 0
    JCity = sum(d)/nY
    
    return JCity

def HydroPower(G, nY):
    
    JH = -1*sum(G)/nY
    
    return JH

def unc_onj_fun(Ju, tr_min, tr_max):
    
    X = np.ndarray(shape = Ju.shape)
    t_mean = (tr_min + tr_max)/2
    
    nO = X.shape[1]
    
    for i in range(0, nO):
        
        X[:, i] = abs(Ju[:, i] - t_mean[i])
    
    return X

def GLUE(GLF, treshold, Y_sim, time, choice):
    
    if choice == 'geq':
        idx  = GLF >= treshold
    else:
        idx = GLF <= treshold
    
    if time == False:
        return(idx)
    else:
    
        if sum(idx) == 0:
            raise Exception('No sample satisfies the condition')
        
        else:
            
            T = Y_sim.shape[1]
            alfa = 0.05
            
            Jb = np.empty(shape = GLF.shape)

            Jb[idx == False] = 0
            Jb[idx] = GLF[idx]
            Jb = Jb/sum(Jb)
            
            Jbt = np.empty(shape = sum(idx))
            Jbt = Jb[idx]
            Llim = np.empty(T)
            Ulim = np.empty(T)
            
            for t in range(0, T):
                idx_sort = np.argsort(Y_sim[idx, t])
                y_sorted = np.sort(Y_sim[idx, t])
                CDF_t = np.cumsum(Jbt[idx_sort])
                Llim_ = y_sorted[CDF_t < alfa]
                
                if any(Llim_):
                    Llim[t] = Llim_[-1]
                else:
                    Llim[t] = y_sorted[0]
                    
                Ulim_ = y_sorted[CDF_t > (1-alfa)]    
                
                if any(Ulim_):
                    Ulim[t] = Ulim_[1]
                else:
                    Ulim[t] = y_sorted[-1]
            
            return idx, Llim, Ulim 

def pawn_sampling(samp_strat, M, distr_fun, distr_par, n, NC):
    
    XX = np.ndarray(shape = (M, n), dtype = 'object')
    xc = np.ndarray(shape = (M), dtype = 'object')
    
    for i in range(0, M):
        
        ifixed = np.tile(False, M)
        ifixed[i] = True
        
        XXi = conditional_sampling(samp_strat, M, distr_fun, distr_par, n, NC, ifixed)
        xc[i] = XXi[0]
        XXi = XXi[1]
        
        for k in range(0, n):
            XX[i, k] = XXi[k]
        
    return XX, xc
        
def conditional_sampling(samp_strat, M, distr_fun, distr_par, n, NC, ifixed):
    
    ivarying = np.tile(True, M)
    ivarying[ifixed] = False
    
    xc = AAT_sampling('lhs', sum(ifixed > 0), distr_fun, distr_par[ifixed],n)
    XX = np.ndarray(shape = (n), dtype = 'object')
    
    for k in range(0, n):
        X_new = AAT_sampling(samp_strat,sum(ivarying>0),distr_fun,distr_par[ivarying],NC)
        X_sub = np.zeros(shape = (NC, M))
        fill = np.ndarray(shape = (NC, 1))*0 + xc[k]
        X_sub[:, ifixed] = fill
        X_sub[:, ivarying] = X_new
        XX[k] = X_sub
        
    return xc, XX

def cdf_support(YU, YC, N):
    
    M = YC.shape[0]
    n = YC.shape[1]
    
    Nsteps = YU.shape[1]
    nV = YU.shape[2]
    
    YF = np.ndarray(shape = (nV, Nsteps), dtype = 'object')
    
    for v in range(0, nV):
        
        Yu = YU[:, :, v]
        Y_min = np.min(Yu, axis = 0)
        Y_max = np.max(Yu, axis = 0)
        
        for i in range(0, M):
            for k in range(0, n):
                
                Yc = YC[i, k][:, :, v]
                Y_min = np.minimum(Y_min, np.min(Yc, axis = 0))
                Y_max = np.maximum(Y_max, np.max(Yc, axis = 0))
                
        
        for t in range(0, Nsteps):
            
            YF[v, t] = np.linspace(Y_min[t], Y_max[t], N)
                
    return YF, Y_min, Y_max

def pawn_ks_timevarying(YU, YC, YF):
    
    M = YC.shape[0]
    n = YC.shape[1]
    
    Nsteps = YU.shape[1]
    nV = YU.shape[2]

    FU = np.ndarray(shape = (nV, Nsteps), dtype = 'object')
    FC = np.ndarray(shape = (nV, Nsteps), dtype = 'object')
    KS = np.ndarray(shape = (nV, M), dtype = 'object')
    KS_max = np.ndarray(shape = (nV), dtype = 'object')
    
    for v in range(0, nV):
        
        for t in range(0, Nsteps):
            
            Yu = YU[:, :, v]
            ecdf = ECDF(Yu[:, t])
            FU[v, t] = ecdf([YF[v, t]])
        
        KS_mi = np.empty(shape = (M, Nsteps))
            
        for i in range(0, M):
            Ksi = np.empty(shape = (n, Nsteps))
            
            for k in range(0, n):
                YCik = YC[i, k]
                YCik = YCik[:, :, v]
                
                for t in range(0, Nsteps):
                    
                    ecdf = ECDF(YCik[:, t])
                    FC[v, t] = ecdf([YF[v, t]])
                    
                    FUt = FU[v, t]
                    FCt = FC[v, t]
                    
                    Ksi[k, t] = max(np.abs(FUt[0, :] - FCt[0, :]))
            
            KS[v, i] = Ksi
            KS_mi[i, :] = np.max(Ksi, axis = 0)
            
        
        KS_max[v] = KS_mi
    
    return KS_max, KS

def pawn_indices(Y, YY, Nboot, v):
    
    '''
    
    M = YY.shape[0]
    n = YY.shape[1]
    
    P = Y.shape[1]
    m = YY[0, 0].shape[1]
    
    if P == m:
        KS_all = np.ndarray(shape = (m), dtype = 'object')
        
        
        for v in range(0, m):
            KS = np.empty(shape = (M, n), dtype = 'object') 
            
            Y_min = min(Y[:, v])
            Y_max = max(Y[:, v])
            
            for i in range(0, M):
                for k in range(0, n):
                    Yc = YY[i, k][:, v]
                    Y_min = min([Y_min, min(Yc)])
                    Y_max = max([Y_max, max(Yc)])
                
            YF = np.linspace(Y_min, Y_max, 3000)
            ecdf = ECDF(Y[:, v])
            FU = ecdf(YF)
        
            for i in range(0, M):
                for k in range(0, n):
                
                    YYik = YY[i, k][:, v]
                    ecdf = ECDF(YYik)
                    FCik = ecdf(YF)
                    KS[i, k] = max(np.abs(FCik - FU))
             
            KS_all[v] = np.max(KS, axis = 1)   
            
            
                
    return KS_all
    '''
    M = YY.shape[0]
    n = YY.shape[1]
    
    NU = Y.shape[0]
    NC = YY[0, 0].shape[0]
    
    alfa = 0.05
    
    if Nboot > 1:
        ks_stat = np.empty(shape = (Nboot, M))
        for j in range(0, Nboot):
            bootsize = NU
            idxi = np.floor(np.random.uniform(size = (bootsize))*NU).astype(int)
            Yb = np.empty(shape = (bootsize, 1))
            Yb[:, 0] = Y[idxi, v] 
            YYb = np.ndarray(shape = (M, n), dtype = 'object')
            for k in range(0, n):
                for i in range(0, M):
                    bootsize = int(0.8*NC)
                    ytmp = YY[i, k][:, v]
                    ytmp2 = np.empty(shape = (bootsize, 1))
                    ytmp2[:, 0] = ytmp[np.random.choice(np.random.permutation(NC), bootsize) ]
                    YYb[i, k] = ytmp2
            
            CDF = pawn_cdfs(Yb, YYb, 0)
            Yfb = CDF[0]
            FUb = CDF[1]
            FCb = CDF[2]
            
            ks = pawn_ks(Yfb, FUb, FCb)
            ks_stat[j, :] = np.max(ks, axis = 0)
        
        T_m = np.mean(ks_stat, axis = 0)
        
        T_lb = np.sort(ks_stat, axis = 0)
        T_lb = T_lb[max(1, int(Nboot*(alfa/2)) ), :]
        T_ub = np.sort(ks_stat, axis = 0)
        T_ub = T_ub[int(Nboot*(1-alfa/2)), :]
        
        return T_m, T_lb, T_ub
    
    else:
        
        CDF = pawn_cdfs(Y, YY, v)
        Yfb = CDF[0]
        FUb = CDF[1]
        FCb = CDF[2]
            
        ks = pawn_ks(Yfb, FUb, FCb)
        ks = np.max(ks, axis = 1)
        
        return ks
        
def pawn_cdfs(Y, YY, v):
    
    M =YY.shape[0]
    n = YY.shape[1]
    
    FC = np.ndarray(shape = (M, n), dtype = 'object')
    
    Y_min = min(Y[:, v])
    Y_max = max(Y[:, v])
    
    for i in range(0, M):
        for k in range(0, n):
            Yc = YY[i, k][:, v]
            Y_min = min([Y_min, min(Yc)])
            Y_max = max([Y_max, max(Yc)])
        
    YF = np.linspace(Y_min, Y_max, 3000)
    ecdf = ECDF(Y[:, v])
    FU = ecdf(YF)

    for i in range(0, M):
        for k in range(0, n):
        
            YYik = YY[i, k][:, v]
            ecdf = ECDF(YYik)
            FCik = ecdf(YF)
            FC[i, k] = FCik
    
    return YF, FU, FC

def pawn_ks(YF, FU, FC):
    
    M = FC.shape[0]
    n = FC.shape[1]
    
    KS = np.empty(shape = (n, M))
    
    for i in range(0, M):
        for k in range(0, n):
            
            FCik = FC[i, k]
            KS[k, i] = max(abs(FU - FCik))
            
    return KS
    

    
    
    
                
                
            



    
    
    
    
    
            
        
        