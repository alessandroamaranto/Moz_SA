#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:25:44 2019

@author: alessandroamaranto
"""

import numpy as np

'''
def storage2level(s, ls_r, A):
    
    if ls_r.size > 0:
        h = np.interp(s, ls_r[2, :], ls_r[0, :])
    else:
        h = s/A
    return h

def level2storage(h, ls_r, A):
    
    if ls_r.size > 0:
        s = np.interp(h, ls_r[0, :], ls_r[2, :])
    else:
        s = h*A
    return s

'''
def level2storage(x):
    
  s = 10**6*(0.018787*x**3 - 1.17037*x**2 + 26.284*x - 209)
  return s

def storage2level(x):
    s = x/(10**6)
    y = 27.06 + 1.136e-01*s -2.696e-04*s**2 + 2.831e-07*s**3
    return y


def integration_daily(HH, tt, s0, uu, uuf, a, cmonth, MEF, EV, evap_d, A):
    
    sim_step = 3600*24/HH;
    
    s = np.zeros(HH + 1)
    r = np.zeros(HH)
    rf = np.zeros(HH)
    
    s[0] = s0
    stor_rel = np.empty(3, 'float')
    
    for i in range(0, HH): #HH

        r[i] = actual_release(uu, s[i], MEF, a)
        rf[i] = actual_release_f(s[i], uuf)
        
        if evap_d == 1:
            S = A
            E = EV[int(cmonth -1)]/1000*S/86400
        else:
            E = 0
        
        s[i + 1] = s[i] + sim_step*(a - r[i] - E - rf[i])
        
    stor_rel[0] = s[HH]
    stor_rel[1] = np.mean(r)
    stor_rel[2] = np.mean(rf)
        
    return stor_rel

def actual_release(u, s, MEF,  a):
    
    qm = min_release(s, MEF, a)
    qM = max_release(s, a)
    
    r = min(qM, max(qm, u))
    return r

def actual_release_f(s, uuf):
    
     h = storage2level(s)
     
     if (h < 28):
         uuf = 0
     
     return uuf

def min_release (s, MEF, a):
    
    h = storage2level(s)
    
    if (h < 28.0):
        q_min = 0
    elif h < 47:
        q_min = MEF
    else:
        q_min = a 
    
    return q_min

def max_release(s, a):
    
    h = storage2level(s)

    if (h < 28.0):
        q_max = 0
    elif h < 47:
        q_max = 4.6*(h - 15.25)**0.5
    else:
        q_max = a
    
    return q_max

def irr_output(theta_d, r, bn, w):
    
    hdg = theta_d[0]
    m = theta_d[1]
    
    hdg_dn = hdg*( bn[1] - bn[0] ) + bn[0]
    
    if(r <= hdg_dn):
        y = np.nanmin(np.array([r, w*(pow(r/hdg_dn,m))]))
    else:
        y = min(r, w)
    return(y)

def energy_production(r, h, qm, hv):
    
    ep = 0.78*0.024*9.8*1000*min(r, qm)*min((h - hv),35)/1000 #MWh

    
    return ep



