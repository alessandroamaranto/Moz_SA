 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:02:11 2019

Emodps implementation of BPL (Mozambique)

@author: alessandroamaranto
"""
import lake_simulation
import DPS_utils
import numpy as np
import pandas as pd
from datetime import datetime
from sys import *
from math import *
from borg import *
from pset import M, N, K
import pickle
import matplotlib.pyplot as plt

nvars = 74
nobjs = 3
nP = 2

bonds = DPS_utils.bounds(M, N, K) 
borg = Borg(nvars, nobjs, 0, lake_simulation.lake)
borg.setBounds(*bonds*nP)
borg.setEpsilons(*[0.01]*nobjs)

start = datetime.now()

result = borg.solve({"maxEvaluations":50000})

print(datetime.now() - start)

theta = DPS_utils.getOut(result, nvars, nobjs)
theta[0].to_csv('Var_1.csv', index = False)
theta[1].to_csv('Obj_1.csv', index = False)

po = theta[1]    

f = open('optim_res.pckl', 'wb')
pickle.dump(theta, f)
f.close()

data = {'a': po.iloc[:, 1],
        'b': po.iloc[:, 2],
        'c': po.iloc[:, 0]}


plt.scatter('a', 'b', c = 'c', data = data)
#plt.show()

