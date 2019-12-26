# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:38:58 2019

@author: Education
"""

import numpy as np
import time
import matplotlib.pyplot as plt
class trajectoryPlanning:
    def __init__(self):
        self.destinatePoint = 1
        self.nowPoint = 0
        self.landmark = np.array([[136,165],[136,123],[275,123],[275,225],[136,225],[136,165]])
        self.newFlag = 1
        self.xhis = [0]
        self.yhis = [0]
        self.speed = 90
    def planning(self):
        if self.destinatePoint < 6:
            if self.newFlag == 1:
                self.planTime = np.linalg.norm(self.landmark[self.destinatePoint]-self.landmark[self.nowPoint])/self.speed
                self.newFlag = 0
                self.beginTime = time.time()
            trajectory = (self.landmark[self.destinatePoint]-self.landmark[self.nowPoint])*(time.time()-self.beginTime)/self.planTime+self.landmark[self.nowPoint]
            if time.time()-self.beginTime>self.planTime:
                self.destinatePoint += 1
                self.nowPoint += 1
                self.newFlag = 1
        else:
            trajectory = self.landmark[0]
        return trajectory 

    
