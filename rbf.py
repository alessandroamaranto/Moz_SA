# -*- coding: utf-8 -*-
"""
Radial basis function utils
"""

import math 
from pset import K, M, N

class Obj(object):
    pass

def setParameters(pTheta):
    
    #global K, N, M
    
    param = []
    cParam = Obj()
    
    cParam.c = []
    cParam.b = []
    cParam.w = []
    
    lin_param = []
    count = 0
    
    for k in range(0, K):
        lin_param.append(pTheta[count])
        count = count + 1
    
    for i in range(0, N):
        for j in range(0, M):
            cParam.c.append(pTheta[count])
            cParam.b.append(pTheta[count + 1])
            count  = count + 2
            
        for k in range(0, K):
            cParam.w.append(pTheta[count])
            count  = count + 1
            
        param.append(cParam)
        cParam = Obj()
        cParam.c = []
        cParam.b = []
        cParam.w = []    
        
    
    X = [lin_param, param]
    return X

def getOutput(inp, pTheta):
    
    p = setParameters(pTheta)
    lin = p[0]
    param = p[1]
    y = []
    phi = []
    
    for j in range(0, N):
        bf = 0
        for i in range(0, M):
            num = (inp[i] - param[j].c[i])*(inp[i] - param[j].c[i])
            den = param[j].b[i]*param[j].b[i]
            
            if (den < 10**(-6)):
                den = 10**(-6)
            bf = bf + num/den
        phi.append(math.exp(-bf))
        
    for k in range(0, K):
         o = lin[k]
         
         for i in range(0, N):
             o = o + param[i].w[k]*phi[i]
         if(o > 1):
             o = 1
         if(o < 0):
             o = 0
         y.append(o)
         
    return(y)
    


 