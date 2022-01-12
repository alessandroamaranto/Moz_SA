#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 17:59:20 2019

@author: alessandroamaranto
"""

import numpy as np
from math import pi
import pset
import utls
import lake_bpl as bpl

def evaluate(*vars):
    
    theta = vars
    idx = pset.N*(2*pset.M + pset.K) + pset.K
    theta_r = theta[:idx]
    theta_d = theta[idx:]
    obj = simulate(theta_r, theta_d)
    
    return(obj)

def simulate(theta_r, theta_d):
    
    H = len(pset.q_in)
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
    
    # Objective function and ts objective
    JJ = []   
    hp  = np.zeros(H)

    for t in range(0, H):
        
        doy[t] = (pset.I[1] + t - 1)%pset.T + 1
        
        inp = np.array([np.sin( 2*pi*doy[t]/pset.T),  
               np.cos(2*pi*doy[t]/pset.T),  
               h[t],
               pset.q_in[t]
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
                                           pset.q_in[t],
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
                bpl.irr_output(theta_d, rd[t + 1], pset.i_bn ,pset.dd[t]),
                max(rd[t + 1] - pset.MEF[t], 0)
                )
        rc[t + 1] = max(rd[t] - rfd[t] - pset.MEF[t], 0)
        
        hp[t] = bpl.energy_production(rd[t + 1], h[t + 1], pset.q_turb, pset.h_down)
        
    rd = rd[1:]
    ru = ru[1:]
    rfd = rfd[1:]
    rc = rc[1:]
    
    '''
    Time-aggregation of the Objectives
    '''
    
    Id = IrrDeficit(rfd, pset.dd, nY)
    Iu = IrrDeficit(ru, pset.du, nY)
    
    JJ.append(
            Id
            )
    
    JJ.append(
            Iu
            )
    
    JJ.append(
            CityDeficit(rc, pset.Cd, nY)
            )
    
    JJ.append(
            HydroPower(hp, nY)
            )
    
    #JJ.append(
    #        env*(365/len(u))
    #        )
    
    return(JJ)

def IrrDeficit(ud, Wd, nY):
    

    d = Wd - ud
    d[d < 0] = 0
    wq = np.quantile(Wd, 0.7)
    
    idx = Wd > wq
    d[idx] = d[idx]*2
    
    Jirr = sum(d)/nY
    
    return Jirr

def CityDeficit(uc, C, nY):
    
    d = C - uc
    d[d < 0] = 0
    JCity = sum(d)/nY
    
    return JCity

def HydroPower(G, nY):
    
    JH = -1*sum(G)/nY
    
    return JH



















    
