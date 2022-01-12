#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 15:30:48 2019

@author: alessandroamaranto
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import utls
#from plotly.offline import plot as py
import matplotlib.pyplot as plt
#import chart_studio.plotly as py



def parallel_dominant_selection(labels, filename, treshold):
    
    x = np.loadtxt(filename)
    
    theta_c = utls.power_equity(filename, treshold)
    
    x = x[:, range(64, 68)]
    xc = x[:, :] 
    idx = np.where(x[:, 1] < treshold)[0]
    x = x[idx, :]

    #x[:, 3]
    
    nS = x.shape[0]
    nO = x.shape[1]
    
    idx = np.ndarray(shape = nO)
    lab = np.zeros(nS)
    
    k = 1
    for i in range(0, nO):
        ind = np.where(x[:, i] == np.min(x[:, i]))[0]
        idx[i] = int(ind[0])
        lab[int(idx[i])] = k
        k = k + 1
    
    lab[theta_c] = nO + 1
    
    x = np.column_stack((x, lab))
    
    x = pd.DataFrame(x, columns = ['i1', 'i2', 'c', 'hp', 'best'])
    
    '''

    
    fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = x['best'],
                   colorscale = [[0,'whitesmoke'],[0.2,'purple'], [0.40, 'blue'], [0.60, 'darkred'], [0.80,'darkorange'], [1.0, 'green']], 
                   
                   ),
        dimensions = list([
            dict(#range = [0,8],
                #constraintrange = [0,200],
                label = 'Irrigation up', values = x['i2']),
            dict(#range = [0,8],
                label = 'Hydropower', values = x['hp']),
            dict(#range = [0,8],
                label = 'City', values = x['c']),
            dict(#range = [0,8],
                label = 'Irrigation down', values = x['i1'])
                ])
                )
                )

    fig.update_layout(
            plot_bgcolor = 'white',
            paper_bgcolor = 'white'
            )   
    py(fig, filename = 'parcoord-dimensions.html')

    fig.show()
    '''
    idx = np.array(x['best'] != 0)
    xs = np.array(x.iloc[idx, :-1])
    lb = np.array(x.iloc[idx, -1])
    
    isD = np.zeros(shape = xc.shape[0])
    
    for i in range(0, xs.shape[0]):
        for j in range(0, xc.shape[0]):
            
            t1 = xs[i, :]
            t2 = xc[j, :]
            
            if (t1 == t2).all():
                isD[j] = True
                
                if lb[i] == nO + 1:
                    isD[j] = 2
                
    return isD

def parallel_plot_colours(filename, idx):
    
    x = np.loadtxt(filename)
    x = x[:, range(64, 68)]
    
    lab = np.zeros(x.shape[0])
    lab[idx] = np.arange(x.shape[1] +2)[1:]
    
    x = np.column_stack((x, lab))
    
    x = pd.DataFrame(x, columns = ['i1', 'i2', 'c', 'hp', 'best'])
    
    fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = x['best'],
                   colorscale = [[0,'whitesmoke'],[0.2,'teal'], [0.40, 'blue'], [0.60, 'darkred'], [0.80,'darkorange'], [1.0, 'green']], 
                   
                   ),
        dimensions = list([
            dict(#range = [0,8],
                #constraintrange = [0,200],
                label = 'Irrigation up', values = x['i2']),
            dict(#range = [0,8],
                label = 'Hydropower', values = x['hp']),
            dict(#range = [0,8],
                label = 'City', values = x['c']),
            dict(#range = [0,8],
                label = 'Irrigation down', values = x['i1'])
                ])
                )
                )

    fig.update_layout(
            plot_bgcolor = 'white',
            paper_bgcolor = 'white'
            )   
    
    savetit = 'parallel_bpl.pdf'
    fig.write_image(savetit)
    
    
            
def plot_cdfs(YF, FU, FCi, ax):
    
    n = FCi.shape[0]
    
    for j in range(0, n):
        ax.plot(YF, FCi[j], 'gray')
    ax.plot(YF, FU, 'r')
    
def plot_bxp(Si, uB, lB, ax, M, labels, title, BW, pn):
    
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    
    purple='#7a0177'; green='#41ab5d'; blue='#1d91c0';yellow='#fdaa09'; red ='#ff0000' #HEX
    colors=[purple,green,blue,yellow,red]
    
    idx1 = np.argmin(BW, axis = 0)
    idx1 = np.where(pn == idx1)
        
    if idx1[0].size == 0:
        idx1 = 4
        col = colors[idx1]
    else:
        col = colors[int(idx1[0])]
    
    
    cpc = []
    bpc = []
    
    dh = 0.25
    dv = 0.005
    
    for j in range(0, M):
        
        rec = Rectangle( (j - dh, Si[j] - dv), 2*dh, 2*dv )

        rec_err = Rectangle( (j - dh, lB[j]), 2*dh, uB[j]-lB[j] )
        bpc.append(rec_err)
        cpc.append(rec)
        
    pc = PatchCollection(cpc, facecolor='black',
                         edgecolor='black')
    
    pc_e = PatchCollection(bpc, facecolor = [col,col, col, col],
                         edgecolor=[col,col, col,col])
    
    ax.add_collection(pc_e)
    ax.add_collection(pc)
    ax.set_xlim([-0.5, M-0.5])
    ax.set_xticks(np.arange(4), minor = False)
    ax.set_xticklabels(labels, fontdict=None, minor=False)
    ax.set_xlabel('Input')
    ax.set_ylabel('Sensitivity Index')
    ax.set_title(title)
    #ax.xticks(np.arange(4), ('Inflow', 'Ag. Demand', 'Inkomati', 'Population'))

def plot_scatters(ax, x, y, idx, lab, BW, pn, i, v):
    
    purple='#7a0177'; green='#41ab5d'; blue='#1d91c0';yellow='#fdaa09'; red ='#ff0000' #HEX
    colors=[purple,green,blue,yellow,red]
    
    idx1 = np.argmin(BW, axis = 0)
    idx1 = np.where(pn == idx1)
    
    yl = ['$\\longleftarrow$ $J_{irr}$ $[m^3/sY]^2$',
          '$\\longleftarrow$ $J_{irr}$ $[m^3/sY]^2$',
          '$\\longleftarrow$ $J_{city}$ $[m^3/sY]$',
          '$\\longleftarrow$ $J_{hp}$ $[GWh/Y]$']
        
    if idx1[0].size == 0:
        idx1 = 4
        col = colors[idx1]
    else:
        col = colors[int(idx1[0])]
    
    gray='#e5e5e5'
    
    x = np.array(x[:])
    y = np.array(y[:])
    
    ax.plot(x, y, c = gray, marker = 'o', linestyle = 'None')
    
    ax.plot(x[idx], y[idx], c = col, marker = 'o', linestyle = 'None')
    ax.set_xlabel(lab)
    
    if i == 0:
        ax.set_ylabel(yl[v])
    
def parallel(x, idx, labels, titolo, typed):
    
    if typed == 'number':
        idx2 = np.zeros(x.shape[0])
        idx2[idx] = 1.0
        idx = idx2[:]
    
    y = np.column_stack((x, idx))
    labels = labels + ['dominant']
    x = pd.DataFrame(y, columns = labels)
    
    fig = go.Figure(data=
    go.Parcoords(
        line = dict(color = x['dominant'],
                   colorscale = [[0,'whitesmoke'],[1.0, 'black']], 
                   
                   ),
        dimensions = list([
            dict(#range = [0,8],
                #constraintrange = [0,200],
                label = labels[0], values = x.iloc[:, 0]#,
                #tickvals = [0.9 , 1]
                ),    
            dict(#range = [0,8],
                label = labels[1], values = x.iloc[:, 1]#,
                #tickvals = [0 , 0.22 ]
                ),
            dict(#range = [0,8],
                label = labels[2], values = x.iloc[:, 2]#,
                #tickvals = [0 , 1.8]
                ),
            dict(#range = [0,8],
                label = labels[3], values = x.iloc[:, 3]#,
                #tickvals = [0 , 0.02 ]
                )
                ])
                )
                )
    #fig.show()
    
    savetit = titolo + '.pdf'
    fig.write_image(savetit)

    
    
    '''
    M = x.shape[1]
    xd = pd.DataFrame(x, columns = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7' ])
    
    lim = np.ndarray(shape = (M, 2))
    
    for i in range(0, M):
        
        lim[i, 0] = np.min(x[idx, i])
        lim[i, 1] = np.max(x[idx, i])
        
    import plotly.graph_objects as go
    
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = 'black',
                       #colorscale = 'Electric',
                       #showscale = True,
                       #cmin = 100,
                       #cmax = 40000
                       ),
            dimensions = list([
                dict(#range = [32000,227900],
                     constraintrange = [lim[0, 0],lim[0, 1]],
                     label = "Input 1", values = xd['I1']),
                dict(#range = [0,700000],
                     constraintrange = [lim[1, 0],lim[1, 1]],
                     label = 'Input 2', values = xd['I2']),
                dict(constraintrange = [lim[2, 0],lim[2, 1]],
                     #tickvals = [0,0.5,1,2,3],
                     #ticktext = ['A','AB','B','Y','Z'],
                     label = 'Input 3', values = xd['I3']),
                dict(constraintrange = [lim[3, 0],lim[3, 1]],#range = [-1,4],
                     #tickvals = [0,1,2,3],
                     label = 'Input 4', values = xd['I4']),
                dict(constraintrange = [lim[4, 0],lim[4, 1]],
                     #range = [134,3154],
                     #visible = True,
                     label = 'Input 5', values = xd['I5']),
                dict(constraintrange = [lim[5, 0],lim[5, 1]],#range = [9,19984],
                     label = 'Input 6', values = xd['I6']),
                dict(constraintrange = [lim[6, 0],lim[6, 1]],#range = [49000,568000],
                     label = 'Input 7', values = xd['I7'])])
        )
    )
    return fig
    '''

def parallel_nor(fig, ax, x, idx, labels, title):
    
    mm = np.min(x, axis = 0)
    M = np.max(x, axis = 0)
    
    X2 = np.ndarray(shape = x.shape)
    
    for i in range(0, X2.shape[1]):
        
        X2[:, i] = (x[:, i] - mm[i])/(M[i] - mm[i])
        
    X2 = X2.transpose()
    
    ax.plot(X2, color = 'gray')
    ax.plot(X2[:, idx], color = 'black')
    ax.set_xlim([-0.5, 3.5])
    ax.set_xticks(np.arange(4), minor = False)
    ax.set_xticklabels(labels, fontdict=None, minor=False)
    ax.set_xlabel('Input')
    ax.set_ylabel('Perturbation range (%)')
    title_ax = 'Behavioral parameters ' + title
    ax.set_title(title_ax)
    
    #title_fig = title + '.pdf'
    #fig.savefig(title_fig)
    
    
    
def tvs_img(x, labels, p_title):
    
    x = x[:,2:]

    nY = x.shape[1]/365
    nV = x.shape[0]
    
    z = np.ndarray(shape = (nV, 365))
    
    for i in range(0, nV):
        
        x[i, :] = np.convolve(x[i, :], np.ones((10,))/10, mode = 'same')
        
        y = x[i, :].reshape(int(nY), 365)
        z[i, :] = np.max(y, axis = 0)
    
    fig, ax = plt.subplots()
    im = ax.imshow(z,  aspect='auto')
    fig.colorbar(im)
    
    
    ax.set_yticks(np.arange(nV))
    ax.set_yticklabels(labels)
    ax.set_title(p_title)
    plt.subplots_adjust(left = 0.18, right = 0.99)
    
    savetit = 'tvsa' + p_title + '.pdf'
    fig.savefig(savetit)

    
    
    
    
    
    

    
        
    
    
    
    
    
    
    