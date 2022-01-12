#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:04:24 2020

@author: alessandroamaranto
"""

import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import pickle
import robustness_utils as rou
import pset
import utls
import lake_bpl as bpl
from math import pi

def robustness_rank(rR, labs, nPol):
    
    rR = normalize_pf(rR, nPol)
    rR = pd.DataFrame(rR, columns = labs) #['IU', 'HP', 'UD', 'ID']
    rR['Name'] = ['IDd', 'IUd', 'UDd', 'HPd', 'NR']
    
    rank_plot(rR, labs, len(labs))

def normalize_pf(x, nP):
    
    x = x/nP
    
    return x
    
def rank_plot(data, labs, nobjs):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    purple='#7a0177'; green='#41ab5d'; blue='#1d91c0';yellow='#fdaa09'; red ='#ff0000' #HEX
    

    colors=[purple,green,blue,yellow,red]
    #linewidths = [6,6,6,6,1]
    parallel_coordinates(data,'Name',color=colors,linewidth=6)

    ax1.set_xticks(np.arange(nobjs))
    ax1.set_xticklabels(['IU', 'HP', 'UD', 'ID'], rotation = 90)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['First', 'Last'], rotation = 90)
    ax1.get_legend().remove()
    plt.text(-.05, 0.5, '$\\longleftarrow$ Direction of Preference', {'color': '#636363', 'fontsize':  12},
             horizontalalignment='left',
             verticalalignment='center',
             rotation=90,
             clip_on=False,
             transform=plt.gca().transAxes)
    
    plt.text(-.15, 0.5, '$\it{minimax}$ Robustness Ranking', {'fontsize':  14},
             horizontalalignment='left',
             verticalalignment='center',
             rotation=90,
             clip_on=False,
             transform=plt.gca().transAxes)

    fig.savefig('minimax_ro_nr.pdf')
    
def pareto_parallel(obj, BW, rank, labs, titolo, nr_idx):
    
    cols = [1, 3, 2, 0]
    obj = obj[:, cols]
    
    mn = np.min(obj, axis = 0)
    mx = np.max(obj, axis = 0)
    
    mn_mx = np.vstack((mn, mx)).reshape(2, 4)
    obj = (obj - mn)/(mx - mn)
    
    r = np.argmin(BW, axis = 0)
    r = np.hstack((r, nr_idx))
    
    cols = [1, 3, 2, 0, 4]
    r = r[cols]
    
    obj_nr = np.delete(obj, r, 0)
    obj_r = obj[r, :]
    #obj_r = obj_r[cols, :]
    
    obj_nr = pd.DataFrame(obj_nr, columns = labs)
    obj_r = pd.DataFrame(obj_r, columns = labs)
    
    names = np.tile(['All Solutions'], obj_nr.shape[0])
    names_r = np.array(['IU', 'HP', 'UD', 'ID', 'NR'])
    
    names = np.hstack((names, names_r))
    
    data = pd.concat([obj_nr, obj_r])
    
    data['Names'] = names
    
    parallel_plots(labs, data, mn_mx, titolo)
    
def parallel_plots(names, data, mn_mx, titolo):
    
    #units=['Deficit [TWh/year]','Deficit [m'r'$^3$/s]'+r'$^2$','[Normalized Squared\nDeficit]','[Bn.USD]']
    nobjs=4
    #policies=5 # number of extreme policies(one for each objective) + compromise policy
    
    mn_mx = pd.DataFrame(mn_mx, columns = names)
    mn_mx.iloc[:, 1] =  mn_mx.iloc[:, 1]*(-1) 
    
    mx=[]
    mn=[]
    for i in range(len(names)):
            mini=str(round(mn_mx[names[i]][1],1))
            maxi=str(round(mn_mx[names[i]][0],1))
            mx.append(maxi)
            mn.append(mini)
        


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #objs1=data['Hydropower'] # here you choose the objective used for the colormap

    gray='#e5e5e5'; purple='#7a0177'; green='#41ab5d'; blue='#1d91c0';yellow='#fdaa09'; red ='#ff0000' #HEX
    

    #### grey scale
    # cmap=truncate_cmap(plt.cm.Greys,n_min=20,n_max=120) # truncate to avoid white and black lines
    # colors=cmap(objs1)

    #find the position where each objective's value is 1
    # l=len(data)

    # colors[l-policies,:]=mpl.to_rgba_array(purple)
    # colors[l-policies+1,:]=mpl.to_rgba_array(yellow)
    # colors[l-policies+2,:]=mpl.to_rgba_array(blue)
    # colors[l-policies+3,:]=mpl.to_rgba_array(green)
    # colors[l-policies+4,:]=mpl.to_rgba_array(pink)
    

    # for a simpler approach with plain gray instead of colormap of grays:
    colors=[gray,green, yellow, blue, purple, red]


    parallel_coordinates(data,'Names',color=colors,linewidth=5)


    #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
     #      ncol=3, mode="expand", borderaxespad=1.5, fontsize=18) 

    i=0
    ax1.set_xticks(np.arange(nobjs))
    ax1.set_xticklabels([mx[i]+'\n'+names[i], mx[i+1]+'\n'+names[i+1],mx[i+2]+'\n'+names[i+2],mx[i+3]+'\n'+names[i+3]],fontsize=20)
    ax2 = ax1.twiny()
    ax2.set_xticks(np.arange(nobjs))
    ax2.set_xticklabels([mn[i], mn[i+1],mn[i+2],mn[i+3]], fontsize=20)
    ax1.get_yaxis().set_visible([])
    ax1.get_legend().remove()
    plt.text(-.05, 0.5, '$\\longleftarrow$ Direction of Preference', {'color': '#636363', 'fontsize':  20},
             horizontalalignment='left',
             verticalalignment='center',
             rotation=90,
             clip_on=False,
             transform=plt.gca().transAxes)
    plt.legend('False') # remove legend

    # plt.colorbar.ColorbarBase(ax=ax, cmap=cmap,
    #                              orientation="vertical")
    fig.set_size_inches(17.5, 9)
    fig.savefig(titolo)
    
def SOW_plot(BB, BW, dim):
    
    r = np.array([np.argmin(BW, axis = 0)[dim], np.argmin(BW, axis = 0)[1] ])#np.argin(BW, axis = 0)[dim]
    sow = np.ndarray(shape = (1, r.shape[0]), dtype = 'object')
    
    for i in range(0, r.shape[0]):
        
        name  = 'policy_2' + str(r[i]) + '.txt'
        r_dim = np.loadtxt(name)
        r_dim = r_dim[:, 2]
        
        N = len(r_dim)
        
        s = np.linspace(min(r_dim), max(r_dim), 1000)
        s = s[np.argsort(-s)]
        
        sw = np.ndarray(s.shape)
        
        for j in range(0, len(s)):
            
            sw[j] = sum(r_dim< s[j])/r_dim.shape[0]
        
        sow[0, i] = np.transpose(np.vstack((s, sw)))
    plt.plot(sow[0, 0][:, 0], sow[0, 0][:, 1])
    plt.plot(sow[0, 1][:, 0], sow[0, 1][:, 1])
    plt.axis([180, 0, 0, 1])
    
    
    s = np.hstack((r_dim[0][:, 2], r_dim[1][:, 2]  ))
    s = np.linspace(min(s), max(s), 100)
    
    s = s[np.argsort(-s)]
    
    for i in range(0, sow.shape[0]):
        
        sow[i, 0] = sum(r_dim[0][:, 2] <= s[i])/N
        sow[i, 1] = sum(r_dim[1][:, 2] <= s[i])/N

    
    plt.plot(sow[:, 1])
    plt.plot(sow[:, 0])
    
    return sow

def probability_plot(BW, filename, nr_idx):
    
    rS = np.argmin(BW, axis = 0)
    rS = np.hstack((rS, nr_idx))
    nP = BW.shape[0]
    nO = BW.shape[1]
    
    gray='#e5e5e5'; purple='#7a0177'; green='#41ab5d'; blue='#1d91c0';yellow='#fdaa09'; red ='#ff0000'
    colors=[purple,green,blue,yellow, red]
    
    fig, ax = plt.subplots(2,2, sharey = True)
    
    rw = [0, 0, 1, 1]
    cl = [0, 1, 0, 1]
    
    mn = (np.zeros(4) + 1) *(9999999)
    mx = (np.zeros(4) + 1) *(-9999999)
    
    for i in range(0, nP):
        
        name  = 'policy_2' + str(i) + '.txt'
        X = np.loadtxt(name)
        
        for j in range(0, nO):
            
            x = X[:, j]
            ecdf = ECDF(x)
            
            z = np.linspace(min(x) , max(x)) 
            if min(z) < mn[j]:
                mn[j] = min(z)
                
            if max(z) > mx[j]:
                mx[j] = max(z)
            
            y = ecdf(z)
            
            r = rw[j]
            c = cl[j]
            ax[r, c].plot(z, y, color = gray,  linewidth=0.5)
            
    for i in rS:
        
        name  = 'policy_2' + str(i) + '.txt'
        X = np.loadtxt(name)
        
        for j in range(0, nO):
            
            x = X[:, j]
            ecdf = ECDF(x)
            
            z = np.linspace(min(x)-0.01 , max(x)) 
                
            y = ecdf(z)
            
            r = rw[j]
            c = cl[j]
            
            if ((rS == i).any()):
                
                idx = np.where(rS == i)[0]
                col = colors[idx[0].astype(int)]
                
                if idx == j:
                    lwd = 4.5
                else:
                    lwd = 1.5
                    
                ax[r, c].plot(z, y, color = col,  linewidth = lwd)
    
    
    
    ax[0, 0].set_ylabel('CDF')
    ax[1, 0].set_ylabel('CDF')
    
    ax[0, 0].set_xticks([mn[0], mx[0]])
    ax[0, 1].set_xticks([mn[1], mx[1]])
    ax[1, 0].set_xticks([mn[2], mx[2]])
    ax[1, 1].set_xticks([mn[3], mx[3]])
    
    
    mx[3] = mx[3]/(-1000)
    mn[3] = mn[3]/(-1000)
    
    mn[0:3] = np.round(mn[0:3], 0)
    mx[0:3] = np.round(mx[0:3], 0)
    
    mn[3] =  np.round(mn[3], 1)
    mx[3] = np.round(mx[3], 1)
    
    ax[0, 0].set_xticklabels([mn[0], mx[0]])
    ax[0, 1].set_xticklabels([mn[1], mx[1]])
    ax[1, 0].set_xticklabels([mn[2], mx[2]])
    ax[1, 1].set_xticklabels([mn[3], mx[3]])
    
    ax[0, 0].set_yticks([0, 1])
    ax[1, 0].set_yticks([0, 1])
    
    ax[0, 0].set_ylim([0, 1])
    ax[1, 0].set_ylim([0, 1])
    
    ax[0, 0].set_xlabel('$\\longleftarrow$ $J_{irr}$ $[m^3/sY]^2$')
    ax[0, 1].set_xlabel('$\\longleftarrow$ $J_{irr}$ $[m^3/sY]^2$')
    ax[1, 0].set_xlabel('$\\longleftarrow$ $J_{city}$ $[m^3/sY]$')
    ax[1, 1].set_xlabel('$\\longleftarrow$ $J_{hp}$ $[GWh/Y]$')
    
    ax[0, 0].set_title('Downstream Irrigation', fontweight='bold')
    ax[0, 1].set_title('Upstream Irrigation', fontweight='bold')
    ax[1, 0].set_title('City', fontweight='bold')
    ax[1, 1].set_title('Hydropower', fontweight='bold')
    
    fig.savefig(filename)
    return(filename)

def probabilistic_level(BW, rob_out, dims, theta, nVars):
    
    with open(rob_out, 'wb') as f:  # Python 3: open(..., 'wb')
         mf, Xq, md, Xd, Xdd, mi, Xi, mcy, Xcy, XU, Ju, RI, BW, BB = pickle.load(f)
    
    rS = np.argmin(BW, axis = 0)[dims]
    policies = theta[rS,:nVars]
    
    nP = policies.shape[0]
    
    h_prob = np.ndarray(shape = (1, nP), dtype = 'object')
    
    for i in range(0, nP):
        policy = policies[i, :]
        h_prob[0, i] = level_simulate(XU, Xq, Xd, Xdd, Xi, Xcy, policy)
    
    
    nY = round(h_prob[0, 1].shape[0]/365, 0)
    doy = np.tile(np.arange(365), int(nY))
    
    ed = h_prob[0, 1].shape[0] - len(doy)
    
    if ed > 0:
        
        ed = np.arange(ed)
        doy = np.hstack((doy, ed))

    
    hY = np.ndarray(shape = (1, nP), dtype = 'object')

    for i in range(0, nP):
        hd = pd.DataFrame(h_prob[0, i])
        hd['doy'] = doy
        
        hday = hd.groupby('doy').mean()
        hY[0, i] = np.transpose(hday.to_numpy())
        
    mn = min(np.min(hY[0, 0]), np.min(hY[0, 1]) )
    mx = max(np.max(hY[0, 0]), np.max(hY[0, 1]) )
    
    bn = np.linspace(mn, mx, 100)   
    z = np.ndarray(shape = (len(bn)-1, hY[0, 0].shape[1]) )
    
    fig, axis = plt.subplots(1, 2, sharey = True)
    
    for i in range(0, nP):
        hy = hY[0, i]
        for j in range(0, hy.shape[1]):
            x = hy[:, j]
            hs = np.histogram(x, bn)
            z[:, j] = np.flip(hs[0])/hy.shape[0]
            
        
        z[z == 0] = float('NaN')
        
        pp = axis[i].imshow(z, cmap = 'Spectral_r', aspect='auto', vmin = 0.0005, vmax = 0.0405)
    
    axis[0].set_yticks((0, len(bn) -1))
    axis[0].set_yticklabels( (round(bn[len(bn)-1], 2), round(bn[0], 2) ), fontdict=None, minor=False)
    axis[0].set_ylabel('z [m]')
    
    axis[0].set_title('Irrigation Upstream', fontweight='bold')
    axis[1].set_title('Hydropower', fontweight='bold')    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    cb = fig.colorbar(pp, cax=cbar_ax)
    cb.set_label('Probability density function')
    
    fig.savefig('probabilisticlevel.pdf')


   


def level_simulate(X, Xq, Xd, Xdd, Xi, Xcy, policy):
    
    for i in range(0, X.shape[1]):
    
        cl = X[:, i]
        mn = np.nanmean(cl)
    
        cl[np.isnan(cl)] = mn
        
        X[:, i] = cl
    
    # Initialize SA out and par
    N = X.shape[0]
    H = Xq.shape[0]
    h_p = np.ndarray(shape = (H + 1, N))
    
    # Initialize policy out and par
    idx = pset.N*(2*pset.M + pset.K) + pset.K
    theta_r = policy[:idx]
       
    for j in range(0, N):
    
        
        # Initialize trajectories
        s = np.zeros(H + 1)
        h = np.zeros(H + 1)
        doy = np.zeros(H)
        
        # Decision variables
        u = np.zeros(H)
        uf = np.zeros(H)
        rd = np.zeros(H + 1)
        ru = np.zeros(H + 1)
        
        # Initial conditions
        h[0] = pset.I[0]
        s[0] = bpl.level2storage(h[0])
        
        # Model forcings
        Q = Xq[:, int(X[j, 0]) ]
        
        
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
            
            
        
            
        
        h_p[:, j] = h
        
        
        
        
    return h_p
           
            
            
            
            
           
            
        
    
    
    
    
    