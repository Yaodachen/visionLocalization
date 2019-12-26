# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 17:53:07 2019

@author: Education
"""
import time
class PIDcontroller:
    def __init__(self,Kp,Ki,Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error = 0
        self.intError = 0
        self.lastTime = time.time()
    def controller(self,newError):
        dt = time.time()-self.lastTime
        self.lastTime = time.time()
        derivateError = (newError-self.error)/dt
        self.intError += newError*dt
        self.error = newError
        output = self.Kp*self.error+self.Ki*self.intError+self.Kd*derivateError
        return output