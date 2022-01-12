#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:35:59 2019

@author: alessandroamaranto
"""

import numpy as np

# Release function parameters
K = 2
M = 4
N = 6
irr = 'yes'

# Input trajectories
q_in = np.loadtxt('Data/PL_qin.txt')
#evap = np.load('Data/PL_evap.txt')
hl = np.loadtxt('Data/PL_h.txt')
MEF = np.loadtxt('Data/PL_mef.txt')

# Objective function inputs
du = np.loadtxt('Data/irr_upstream.txt')[0:-1]
dd = np.loadtxt('Data/irr_downstream.txt')[0:-1]
Cd = 2.5
q_turb = 6
h_down = 12

# Initial conditions and parameters
I = [42.387, 1]
Is = 24
T = 365
A = -999

# Input normalization params 
bn = [np.array([-1, -1, 33.24, 0]),
      np.array([1, 1, 47.24, 311])]

r_max = 5615
irr_max = 1.5

i_bn = [0, 10]

# Irrigated areas (in case of perturbing)
hau = 2400.1563646226728
had = 1004.131282992885
