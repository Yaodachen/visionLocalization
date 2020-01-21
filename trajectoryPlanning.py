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
        self.landmark = np.array([[140,215],[30,225],[30,33],[265,43],[265,215],[200,215],[180,215]])
        self.destinatePoint = self.nowPoint = len(self.landmark)
        self.newFlag = 1
        self.xhis = [0]
        self.yhis = [0]
        self.speed = 50
        self.end = True
    def planning(self,car_location = None):
        if self.destinatePoint < len(self.landmark):
            if self.newFlag == 1:
                # if self.nowPoint == 2:
                #     self.speed = 30
                # elif self.nowPoint == 3:
                #     self.speed = 70
                # else:
                #     self.speed = 50
                self.planTime = np.linalg.norm(self.landmark[self.destinatePoint]-self.landmark[self.nowPoint])/self.speed
                self.newFlag = 0
                self.beginTime = time.time()
            trajectory = (self.landmark[self.destinatePoint]-self.landmark[self.nowPoint])*(time.time()-self.beginTime)/self.planTime+self.landmark[self.nowPoint]
            if time.time()-self.beginTime>self.planTime:
                self.destinatePoint += 1
                self.nowPoint += 1
                self.newFlag = 1
        else:
            trajectory = self.landmark[len(self.landmark)-1]
            self.end = True
        return self.end,trajectory

    def start(self):
        self.destinatePoint = 1
        self.nowPoint = 0
        self.newFlag = 1
        self.end = False

class trajectoryPlanning_fast:
    def __init__(self):
        self.landmark = np.array([[140,235],[20,240],[30,33],[285,43],[285,225],[280,240],[180,240],[20,230]]) #np.array([[180,235],[130,235],[136,103],[268,113],[268,235],[200,240],[180,235]])
        self.destinatePoint = self.nowPoint = len(self.landmark)
        self.newFlag = 1
        self.xhis = [0]
        self.yhis = [0]
        self.speed = 50
        self.end = True
    def planning(self):
        if self.destinatePoint < len(self.landmark):
            if self.newFlag == 1:
                if self.nowPoint > 2:
                    self.speed = 100
                elif self.nowPoint == 0:
                    self.speed = 48
                elif self.nowPoint == 6:
                    self.speed = 0
                else:
                    self.speed = 50
                self.planTime = np.linalg.norm(self.landmark[self.destinatePoint]-self.landmark[self.nowPoint]) / self.speed
                self.newFlag = 0
                self.beginTime = time.time()
            trajectory = (self.landmark[self.destinatePoint]-self.landmark[self.nowPoint])*(time.time()-self.beginTime)/self.planTime+self.landmark[self.nowPoint]
            if time.time()-self.beginTime>self.planTime:
                self.destinatePoint += 1
                self.nowPoint += 1
                self.newFlag = 1
        else:
            trajectory = self.landmark[len(self.landmark)-1]
            self.end = True
        return self.end,trajectory

    def start(self):
        self.destinatePoint = 1
        self.nowPoint = 0
        self.newFlag = 1
        self.end = False


class trajectoryPlanning_part:
    def __init__(self):
        self.landmark = np.array([[255,235],[136,235],[35,235],[35,125]])
        self.destinatePoint = self.nowPoint = 999
        self.newFlag = 1
        self.xhis = [0]
        self.yhis = [0]
        self.speed = 50
        self.beginTime = time.time()
        self.end = True

    def planning(self):
        if self.destinatePoint < len(self.landmark):
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
            trajectory = self.landmark[len(self.landmark)-1]
            self.end = True
        return self.end,trajectory

    def start(self):
        self.destinatePoint = 1
        self.nowPoint = 0
        self.newFlag = 1
        self.end = False

class lap1:
    def __init__(self):
        self.landmark = np.array([[180,235],[150,225],[150,25],[275,35],[275,235],[200,235],[180,235]])
        self.destinatePoint = self.nowPoint = len(self.landmark)
        self.newFlag = 1
        self.xhis = [0]
        self.yhis = [0]
        self.speed = 50
        self.end = True
    def planning(self,car_location = None):
        if self.destinatePoint < len(self.landmark):
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
            trajectory = self.landmark[len(self.landmark) - 1]
            trajectory = self.landmark[len(self.landmark) - 1]
            self.end = True
        return self.end,trajectory

    def start(self):
        self.destinatePoint = 1
        self.nowPoint = 0
        self.newFlag = 1
        self.end = False

class lap2:
    def __init__(self):
        self.landmark = np.array([[100,235],[130,225],[130,35],[20,25],[20,235],[60,235],[100,235]])
        self.destinatePoint = self.nowPoint = len(self.landmark)
        self.newFlag = 1
        self.xhis = [0]
        self.yhis = [0]
        self.speed = 50
        self.end = True
    def planning(self,car_location = None):
        if self.destinatePoint < len(self.landmark):
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
            trajectory = self.landmark[len(self.landmark) - 1]
            self.destinatePoint = 1
            self.nowPoint = 0
            self.newFlag = 1
        return self.end,trajectory
    def start(self):
        self.destinatePoint = 1
        self.nowPoint = 0
        self.newFlag = 1
        self.end = False