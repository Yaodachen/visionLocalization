# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:36:56 2019

@author: Education
"""
import numpy as np
def angleWeightAverage(a,b,ka,kb):
    if np.abs(a-b)<np.pi:
        c = ka*a+kb*b
    else:
        c = ka*(a+2*np.pi*(a<0))+kb*(b+2*np.pi*(b<0))
    if c>np.pi:
        c = c-2*np.pi
    return c
K = 0.1
a = 170/180*np.pi
b = -175/180*np.pi
c = 12/180*np.pi
d = angleWeightAverage(a,b,1-K,K)+c
if d > np.pi:
    d -= 2*np.pi
elif d < -np.pi:
    d += 2*np.pi
print(d/np.pi*180)