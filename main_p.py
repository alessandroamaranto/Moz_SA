#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:15:34 2019

@author: alessandroamaranto
"""

 #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:02:11 2019

Emodps implementation in Pequenos Libombos

@author: alessandroamaranto
"""

import model_bpl as mlc
import utls
from sys import *
from math import *
from borg import *
from pset import M, N, K, irr
import os

def Optimization(j):
    
    # Optimization settings
    nvars = 64
    nobjs = 4
    nP = 1
    nfeval = 500000
    bonds = utls.bounds(M, N, K, irr) 
        
    # Where to save seed and runtime files
    os_fold = utls.Op_Sys_Folder_Operator()  # Folder operator for operating system
    main_output_file_dir = os.getcwd() + os_fold + 'out'
    output_location = main_output_file_dir + os_fold + 'sets'
    runtime_freq = 50000
    eps = [1, 4, 0.3, 20]
    
    runtime_filename = main_output_file_dir + os_fold + 'runtime_file_seed_' + str(j+1) + '.runtime'
    borg = Borg(nvars, nobjs, 0, mlc.evaluate)
    borg.setBounds(*bonds*nP)
    borg.setEpsilons(*eps)
       
    result = borg.solve({"maxEvaluations": nfeval, "runtimeformat": 'borg', "frequency": runtime_freq,
     "runtimefile": runtime_filename})
    
    
    if result:
    # This particular seed is now finished being run in parallel. The result will only be returned from
    # one node in case running Master-Slave Borg.
    #result.display()
     
    # Create/write objective values and decision variable values to files in folder "sets", 1 file per seed.
        f = open(output_location + os_fold + 'Borg_DPS_PySedSim' + str(j+1) + '.set', 'w')
        f.write('#Borg Optimization Results\n')
        f.write('#First ' + str(nvars) + ' are the decision variables, ' + 'last ' + str(nobjs) +
                ' are the ' + 'objective values\n')
        for solution in result:
            line = ''
            for i in range(len(solution.getVariables())):
                line = line + (str(solution.getVariables()[i])) + ' '
     
            for i in range(len(solution.getObjectives())):
                line = line + (str(solution.getObjectives()[i])) + ' '
     
            f.write(line[0:-1]+'\n')
        f.write("#")
        f.close()
     
    # Create/write only objective values to files in folder "sets", 1 file per seed. Purpose is so that
    # the file can be processed in MOEAFramework, where performance metrics may be evaluated across seeds.
    f2 = open(output_location + os_fold + 'Borg_DPS_PySedSim_no_vars' + str(j+1) + '.set', 'w')
    for solution in result:
        line = ''
        for i in range(len(solution.getObjectives())):
            line = line + (str(solution.getObjectives()[i])) + ' '
     
        f2.write(line[0:-1]+'\n')
    f2.write("#")
    f2.close()
         
            #print("Seed %s complete") %j
            
        
        
            #result = borg.solve({"maxEvaluations": nfeval, "runtimeformat": 'borg', "frequency": runtime_freq,
            #                     "runtimefile": runtime_filename})

#theta = utls.getOptRes(result, nvars, nobjs)



