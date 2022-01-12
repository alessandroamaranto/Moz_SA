#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:21:36 2019

@author: alessandroamaranto
"""
''' 
Select optimal solution per each stakeholder + compromise solution


idx_all = sag.parallel_dominant_selection(-999, filename, 400)
idx_equity = sag.parallel_dominant_selection(-999, filename, 150)

idx = np.column_stack((np.where(idx_all == 1), np.where(idx_equity == 2)))

sag.parallel_plot_colours(filename, idx[0])
#idx = np.where(idx == 1)

theta = np.loadtxt(filename)
theta_s = theta[idx[0], :]

obj = theta_s[:, range(64, 68)]
policies = theta_s[:, range(0, 64)] 

idx = np.array([np.where(obj[:, 1] == np.min(obj[:, 1]))[0][0] ,
                np.where(obj[:, 3] == np.min(obj[:, 3]))[0][0],
                obj.shape[0] - 1]
)

nOut = idx.shape[0]
tresholds = np.max(theta[:, range(64, 68)], axis = 0)
'''

import numpy as np
import tvsa_pawn_rob as tvsa
import robustness_utils as rob_u
import robustness_graphics as rob_g

'''
General settings
'''

filename = 'out/sets/Borg_DPS_PySedSim37.set'

theta = np.loadtxt(filename)
nvars = 64
nobjs = 4
obj = theta[:, nvars:]

'''
Select and plot robust solutions 
'''

RI = rob_u.robustness(filename, 64, 4)
BW = RI[1]; BB = RI[2]; RI = RI[0]
nr_idx = rob_u.non_robust_idx(BW)
rS = rob_u.select_robust_solutions(BW, nr_idx)


'''
Plot robustness output
'''

rob_g.robustness_rank(rS, ['IU', 'HP', 'UD', 'ID'], obj.shape[0] -1)
rob_g.pareto_parallel(obj, BW, rS, ['IU', 'HP', 'UD', 'ID'], 'BW_rob_nr.pdf', nr_idx)
rob_g.probability_plot(BW, 'probabilities.pdf')

'''
Uncertainty and Sensitivity Analysis

'''

SA_out = np.ndarray(shape = rS, dtype = 'object')
pn_all = np.hstack(np.argmin(BW, axis = 0), nr_idx)

for pn in pn_all:
    tvsa.pawnSA(nvars, nobjs, 'Robustness.pkl', filename, pn)
